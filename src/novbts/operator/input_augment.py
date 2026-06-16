#!/usr/bin/env python3
"""A/B test: does feeding the analytic Cattaneo-Mindlin field as extra INPUT
channels break the tangential ceiling? (input-representation hypothesis)

Diagnosis (field2field.params_to_fieldinput): the 3 input channels are
  ch0 pen   = Hertz-shaped penetration  -> rich structure, normal predicts well
  ch1 sx*mask, ch2 sy*mask = FLAT top-hats -> no stick-slip annulus, no
       stick->slip boundary. The net must invent the entire radial stick-slip
       structure from a constant patch. The "normal good / tangential bad"
       asymmetry mirrors this "structured channel / flat channel" asymmetry.

Fix (option 1+3): append the analytic Hertz-Mindlin displacement field
(ux_t with stick-core/slip-annulus, uy_t, uz) as input channels so the FNO only
learns the FEM *residual* on the Mindlin prior -- the same mechanism that makes
the normal channel work. No new data; reuse the existing swept FEM GT.

  python -m novbts.operator.input_augment --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, train_operator, predict_raw, rel_l2_per_mode, tangential_dir_error,
    count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import (
    MODE_NAMES, hertz_mindlin_field, hertz_scalars, mindlin_stick_radius,
)
from novbts.paths import FEM, RUNS, ensure


def boundary_channels(params, coords, side):
    """Option 2: explicit SHARP stick-slip boundary cues FNO's spectral conv
    cannot synthesize. Returns [N,2,H,W]: (1) radial coord r/a_eff, (2) stick-core
    mask (r < Mindlin stick radius c). Both are sharp/located, unlike a top-hat."""
    p = np.asarray(params, dtype=np.float64)
    x0, y0, depth, R, sx, sy, mu, E, geom = [p[:, i] for i in range(9)]
    X = coords[:, 0].reshape(side, side); Y = coords[:, 1].reshape(side, side)
    r = np.sqrt((X[None] - x0[:, None, None]) ** 2 + (Y[None] - y0[:, None, None]) ** 2 + 1e-12)
    a, P, p0, C = hertz_scalars(depth, R, E)
    a_eff = np.where(geom > 0.5, R, a)[:, None, None]
    g = np.sqrt(sx ** 2 + sy ** 2) / np.clip(mu, 1e-6, None)
    c = mindlin_stick_radius(np.where(geom > 0.5, R, a), g)[:, None, None]
    r_norm = np.clip(r / np.clip(a_eff, 1e-6, None), 0.0, 3.0)
    stick_mask = (r < c).astype(np.float32)
    return np.stack([r_norm, stick_mask], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    coords = np.load(args.data, allow_pickle=True)["coords"]
    print(f"device={DEV}  input-augment A/B  data={args.data}  N={N} side={side}")

    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    inp3, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))

    # analytic Cattaneo-Mindlin field (stick-core/slip-annulus structure) -> 3 channels
    disp_a, _ = hertz_mindlin_field(D["params"], coords)
    af = torch.tensor(disp_a.reshape(-1, side, side, 3).transpose(0, 3, 1, 2)).to(DEV)
    bc = torch.tensor(boundary_channels(D["params"], coords, side)).to(DEV)   # [N,2,H,W]
    inp6 = torch.cat([inp3, af], 1)                  # opt1: + analytic Mindlin field
    inp5 = torch.cat([inp3, bc], 1)                  # opt2: + sharp boundary cues
    inp8 = torch.cat([inp3, af, bc], 1)              # opt1+2

    tr = torch.arange(0, N - nt, device=DEV)
    te = torch.arange(N - nt, N, device=DEV)

    def run(tag, inp, in_ch):
        im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
        nin = lambda t: (t - im) / istd
        nout = lambda t: (t - om) / ostd
        nsc = lambda t: (t - sm) / sstd
        torch.manual_seed(0)
        m = FNOField(in_ch=in_ch, modes=args.modes).to(DEV)
        secs, _ = train_operator(m, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]),
                                 mode[tr], cg, args.epochs, args.lr)
        pred = predict_raw(m, nin(inp[te]), nsc(scal[te]), cg, ostd, om)
        r = {"relative_l2": rel_l2_per_mode(pred, out[te], mode[te]),
             "tangential_dir_error_deg": tangential_dir_error(pred, out[te], mode[te]),
             "params": count_parameters(m), "train_s": round(secs, 1)}
        L = r["relative_l2"]
        print(f"[{tag:16s}] overall={L['overall']:.3f}  normal={L['normal']:.3f}  "
              f"stick={L['stick']:.3f}  partial={L['partial_slip']:.3f}  "
              f"full={L['full_slip']:.3f}  tang_dir={r['tangential_dir_error_deg']:.1f}°")
        return r

    variants = [
        ("baseline_3ch", inp3, 3),
        ("opt1_mindlin_6ch", inp6, 6),
        ("opt2_boundary_5ch", inp5, 5),
        ("opt1+2_8ch", inp8, 8),
    ]
    results = {tag: run(tag, inp, ch) for tag, inp, ch in variants}
    print("\n=== input-representation A/B (rel L2 on FEM GT, lower=better) ===")
    print(f"{'variant':18s} {'overall':>8s} {'partial':>8s} {'full_slip':>10s} {'tang_dir°':>10s}")
    base = results["baseline_3ch"]
    for tag, _, _ in variants:
        L = results[tag]["relative_l2"]
        print(f"{tag:18s} {L['overall']:8.3f} {L['partial_slip']:8.3f} {L['full_slip']:10.3f} "
              f"{results[tag]['tangential_dir_error_deg']:10.1f}")
    print(f"\n(baseline tang_dir {base['tangential_dir_error_deg']:.1f}°, full_slip {base['relative_l2']['full_slip']:.3f})")

    phase_dir = RUNS / "phase3_fem"; ensure(phase_dir)
    out_path = phase_dir / "input_augment.json"
    json.dump({"gt": args.data, "models": results}, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
