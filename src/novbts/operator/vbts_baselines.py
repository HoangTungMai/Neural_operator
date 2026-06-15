#!/usr/bin/env python3
"""Apples-to-apples bake-off: our neural operator (FNO) vs the marker-motion
models of representative VBTS (vision-based tactile sensor) simulators, all
trained and evaluated on the SAME PhysX-FEM ground truth.

Why reimplement rather than cite published numbers: cross-paper accuracy/fps is
not comparable (different sensors, output modalities -- RGB image vs our marker
displacement field, different contacts). The fair test is to give each method's
*marker-motion model* the same FEM training data and the same metric.

Representative VBTS simulators and the marker-motion model we reimplement:
  - TACTO  (Wang et al., RA-L 2022): physics/rendering sim on PyBullet. Its
    marker motion is KINEMATIC -- markers follow the contact surface; tangential
    motion is a rigid drag of the contact patch, with NO friction / stick-slip
    model. -> `TactoKinematic`.
  - Taxim  (Si & Yuan, RA-L 2022) and FOTS (Zhao et al., 2023): example-based /
    fast optical sims whose marker field is a LINEAR ELASTIC SUPERPOSITION model
    (a calibrated linear/Green's-function map from contact load to marker
    displacement). -> `LinearSuperposition` (a single linear, shift-invariant
    conv = the discretised superposition kernel, no nonlinearity).
  - per-point MLP: learned-but-LOCAL lower bound (no global context).
  - FNO (ours): non-local spectral operator.

We reimplement the marker-motion CORE of each simulator (not the optical
renderer) and fit its free parameters on our FEM train split, exactly like we
train the operators. If FNO beats the linear superposition model, that isolates
the value of modelling the *nonlinear, non-local* stick-slip field that the
linear/kinematic VBTS sims cannot represent.

Usage:
  python -m novbts.operator.vbts_baselines --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from novbts.operator.field2field import (
    FNOField, PerPointMLP, train_operator, predict_raw, throughput,
    rel_l2_per_mode, tangential_dir_error, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import MODE_NAMES, hertz_mindlin_field
from novbts.paths import FEM, RUNS, ensure


class TactoKinematic(nn.Module):
    """TACTO-style marker motion: kinematic and friction-less.

    uz (out-of-plane) <- penetration; in-plane <- (a) rigid shear drag of the
    contact patch + (b) radial push by the penetration surface-slope. There is
    NO stick-slip model: the in-plane field is a uniform rigid drag, missing the
    stuck-centre / slipping-annulus structure FEM produces. The few free scalars
    are fit by the same MSE training loop as the operators.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))   # uz   <- penetration
        self.b = nn.Parameter(torch.tensor(0.0))   # in-plane <- penetration gradient (radial)
        self.g = nn.Parameter(torch.tensor(1.0))   # in-plane <- rigid shear drag

    def forward(self, field, scal):
        pen = field[:, 0:1]
        sx, sy = field[:, 1:2], field[:, 2:3]
        gx = torch.zeros_like(pen); gy = torch.zeros_like(pen)
        gx[..., :, 1:-1] = (pen[..., :, 2:] - pen[..., :, :-2]) * 0.5
        gy[..., 1:-1, :] = (pen[..., 2:, :] - pen[..., :-2, :]) * 0.5
        ux = self.g * sx + self.b * gx
        uy = self.g * sy + self.b * gy
        uz = self.a * pen
        return torch.cat([ux, uy, uz], 1)


class LinearSuperposition(nn.Module):
    """Taxim/FOTS-style linear elastic superposition marker model.

    output field = (input field + broadcast mu,E) convolved with a single learned
    linear, shift-invariant kernel -- the discretised elastic Green's function.
    No activation => the map is exactly linear in the contact input, so it cannot
    represent the nonlinear stick->partial->full slip transition.
    """
    def __init__(self, ksize=31, in_ch=5, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, padding=ksize // 2, bias=True)

    def forward(self, field, scal):
        b, c, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        return self.conv(torch.cat([field, sc], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ksize", type=int, default=31)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    print(f"device={DEV}  VBTS bake-off  data={args.data}  N={N} side={side}  test=last {nt}")
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))

    tr = torch.arange(0, N - nt, device=DEV)
    te = torch.arange(N - nt, N, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd

    def fit(model, is_mlp=False):
        torch.manual_seed(0)
        model = model.to(DEV)
        secs, _ = train_operator(model, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]),
                                 mode[tr], cg, args.epochs, args.lr, is_mlp=is_mlp)
        return model, secs

    def evaluate(model, is_mlp=False):
        pred = predict_raw(model, nin(inp[te]), nsc(scal[te]), cg, ostd, om, is_mlp=is_mlp)
        return {"relative_l2": rel_l2_per_mode(pred, out[te], mode[te]),
                "tangential_dir_error_deg": tangential_dir_error(pred, out[te], mode[te]),
                "throughput_fps": throughput(model, nin(inp[te]), nsc(scal[te]), cg, is_mlp=is_mlp),
                "params": count_parameters(model)}

    # representative VBTS marker-motion models + our operator (+ MLP lower bound)
    specs = [
        ("tacto_kinematic", lambda: TactoKinematic(), False,
         "TACTO-style: kinematic, friction-less (rigid shear drag, no stick-slip)"),
        ("taxim_fots_linear", lambda: LinearSuperposition(args.ksize), False,
         "Taxim/FOTS-style: linear elastic superposition (one linear conv kernel)"),
        ("mlp_perpoint", lambda: PerPointMLP(), True,
         "per-point MLP: learned but local (lower bound)"),
        ("fno_ours", lambda: FNOField(modes=args.modes), False,
         "ours: non-local spectral neural operator"),
    ]

    results = {}
    for key, ctor, is_mlp, desc in specs:
        model, secs = fit(ctor(), is_mlp=is_mlp)
        r = evaluate(model, is_mlp=is_mlp)
        r["train_s"] = round(secs, 1)
        r["desc"] = desc
        results[key] = r
        print(f"[{key:18s}] trained {secs:5.0f}s  params={r['params']:>9d}  "
              f"overall={r['relative_l2']['overall']:.3f}  tang_dir={r['tangential_dir_error_deg']:5.1f}°")

    # ---- first-principles physics baseline: Cattaneo-Mindlin analytic contact model ----
    # The textbook contact-mechanics field (Hertz normal + Cattaneo-Mindlin partial
    # slip) underlying analytical tactile sims. Its amplitude is in its own units, so
    # we give it the FAIREST shot: a per-channel affine (scale+bias) fit by least
    # squares on the train split -> isolates whether the analytic field STRUCTURE
    # (not amplitude) matches FEM, i.e. what the neural operator adds over textbook physics.
    coords_np = np.load(args.data, allow_pickle=True)["coords"]
    params_np = D["params"] if isinstance(D["params"], np.ndarray) else np.asarray(D["params"])
    disp_a, _ = hertz_mindlin_field(params_np, coords_np)            # [N,M,3]
    af = torch.tensor(disp_a.reshape(-1, side, side, 3).transpose(0, 3, 1, 2)).to(DEV)
    tr_np, te_np = tr.cpu().numpy(), te.cpu().numpy()
    pred_a = torch.zeros_like(af[te])
    for c in range(3):                                               # per-channel scale+bias
        A = af[tr, c].reshape(-1); Y = out[tr, c].reshape(-1)
        M = torch.stack([A, torch.ones_like(A)], 1)
        wb = torch.linalg.lstsq(M, Y.unsqueeze(1)).solution.squeeze(1)
        pred_a[:, c] = wb[0] * af[te, c] + wb[1]
    import time as _t
    _ = hertz_mindlin_field(params_np[te_np], coords_np)             # warm
    t0 = _t.perf_counter()
    for _ in range(5):
        hertz_mindlin_field(params_np[te_np], coords_np)
    cm_fps = (5 * len(te_np)) / (_t.perf_counter() - t0)
    results["cattaneo_mindlin_analytic"] = {
        "relative_l2": rel_l2_per_mode(pred_a, out[te], mode[te]),
        "tangential_dir_error_deg": tangential_dir_error(pred_a, out[te], mode[te]),
        "throughput_fps": cm_fps, "params": 6,
        "desc": "first-principles Hertz + Cattaneo-Mindlin analytic, per-channel affine-calibrated to FEM"}
    rc = results["cattaneo_mindlin_analytic"]
    print(f"[cattaneo_mindlin   ] analytic (calibrated)  overall={rc['relative_l2']['overall']:.3f}  "
          f"tang_dir={rc['tangential_dir_error_deg']:5.1f}°")

    fno = results["fno_ours"]["relative_l2"]["overall"]
    summary = {
        "gt": args.data, "side": side, "train_frames": int(N - nt), "test_frames": nt,
        "note": ("VBTS-simulator marker-motion models reimplemented and fit on our FEM GT. "
                 "TACTO=kinematic/friction-less; Taxim/FOTS=linear elastic superposition; "
                 "cattaneo_mindlin=first-principles analytic contact physics (affine-calibrated); "
                 "MLP=local lower bound; FNO=ours (non-local). Cross-paper numbers are not "
                 "comparable (RGB image vs marker field), hence this in-house bake-off. Running "
                 "the originals' pre-calibrated weights on our gel would be wrong-sensor, not "
                 "fairer; the fair route-A is fitting each model's form to our data, as done here."),
        "models": results,
        "fno_advantage_x": {k: round(results[k]["relative_l2"]["overall"] / fno, 2)
                            for k in results if k != "fno_ours"},
    }
    phase_dir = RUNS / "phase3_fem"
    ensure(phase_dir)
    out_path = phase_dir / "vbts_baselines.json"
    json.dump(summary, open(out_path, "w"), indent=2)

    print("\n=== VBTS marker-motion bake-off (rel L2 on FEM GT, lower=better) ===")
    print(f"{'method':20s} {'overall':>8s} {'normal':>7s} {'stick':>7s} {'partial':>8s} "
          f"{'full':>6s} {'tang°':>6s} {'fps':>7s} {'params':>9s}")
    order = ["tacto_kinematic", "cattaneo_mindlin_analytic", "taxim_fots_linear", "mlp_perpoint", "fno_ours"]
    for k in order:
        r = results[k]; L = r["relative_l2"]
        print(f"{k:20s} {L['overall']:8.3f} {L['normal']:7.3f} {L['stick']:7.3f} "
              f"{L['partial_slip']:8.3f} {L['full_slip']:6.3f} "
              f"{r['tangential_dir_error_deg']:6.1f} {r['throughput_fps']:7.0f} {r['params']:9d}")
    print("\nFNO advantage (their overall rel L2 / ours):")
    for k, v in summary["fno_advantage_x"].items():
        print(f"  {k:20s} {v:.2f}x")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
