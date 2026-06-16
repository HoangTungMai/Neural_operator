#!/usr/bin/env python3
"""Phase 4 (de-risk probe): does the trained FNO capture a CONTROLLABLE slip lever?

The anti-slip policy task only makes sense if, holding (mu,R,E) fixed, the FNO's
predicted slip indicator responds monotonically to the action (normal depth) and
to the tangential disturbance |shear|. This probe trains the forward FNO once and
sweeps each axis, printing the slip-proxy curve -- so we validate the lever on the
surrogate (and sanity-check vs the analytic physics trend) BEFORE building the
policy trainer.

slip proxy = mean tangential displacement magnitude over the contact mask
             (full slip -> contact translates a lot; stick -> little net motion).

  python -m novbts.operator.diff_policy --data data/fem/shear_fine_swept_normaug.npz --probe
"""
import argparse
import json

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, train_operator, params_to_fieldinput, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import hertz_mindlin_field
from novbts.paths import FEM, RUNS, ensure


def contact_mask(params, coords, side):
    """[N,H,W] mask r <= contact radius (sphere: Hertz a; flat: R)."""
    from novbts.groundtruth.hertz_mindlin import hertz_scalars
    p = np.asarray(params, dtype=np.float64)
    x0, y0, depth, R, sx, sy, mu, E, geom = [p[:, i] for i in range(9)]
    X = coords[:, 0].reshape(side, side); Y = coords[:, 1].reshape(side, side)
    r = np.sqrt((X[None] - x0[:, None, None]) ** 2 + (Y[None] - y0[:, None, None]) ** 2 + 1e-12)
    a, _, _, _ = hertz_scalars(depth, R, E)
    a_eff = np.where(geom > 0.5, R, a)[:, None, None]
    return (r <= a_eff).astype(np.float32)


def slip_proxy(field, mask):
    """field [N,3,H,W] (ux,uy,uz), mask [N,H,W] -> [N] mean |u_tangential| over contact."""
    tmag = torch.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2 + 1e-12)   # [N,H,W]
    m = mask
    return (tmag * m).sum((1, 2)) / (m.sum((1, 2)) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sweep-n", type=int, default=12)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    coords = np.load(args.data, allow_pickle=True)["coords"]
    P = D["params"]
    print(f"device={DEV}  phase4 probe  data={args.data}  N={N} side={side}")

    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    tr = torch.arange(0, N - nt, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nsc = lambda t: (t - sm) / sstd
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)

    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    secs, _ = train_operator(fno, nin(inp[tr]), (out[tr] - om) / ostd, nsc(scal[tr]),
                             mode[tr], cg, args.epochs, args.lr)
    fno.eval()
    print(f"[FNO] trained {secs:.0f}s  ({count_parameters(fno)} params)\n")

    @torch.no_grad()
    def fno_proxy(rows):
        inp_np, scal_np = params_to_fieldinput(rows, coords, side)
        inp_t = torch.tensor(inp_np, device=DEV); scal_t = torch.tensor(scal_np, device=DEV)
        pred = fno(nin(inp_t), nsc(scal_t)) * ostd + om
        mask = torch.tensor(contact_mask(rows, coords, side), device=DEV)
        return slip_proxy(pred, mask).cpu().numpy()

    def gt_proxy(rows):
        disp, _ = hertz_mindlin_field(rows, coords)        # [N,M,3] analytic reference
        f = torch.tensor(disp.reshape(-1, side, side, 3).transpose(0, 3, 1, 2).astype(np.float32))
        mask = torch.tensor(contact_mask(rows, coords, side))
        return slip_proxy(f, mask).numpy()

    # base context = median of training params; force a slip-inducing shear
    base = np.median(P[:N - nt], 0)
    R_med, mu_med, E_med = base[3], base[6], base[7]
    d_lo, d_hi = np.quantile(P[:, 2], 0.05), np.quantile(P[:, 2], 0.95)
    s_lo, s_hi = 0.0, float(np.quantile(np.hypot(P[:, 4], P[:, 5]), 0.95))
    print(f"context: R={R_med*1e3:.1f}mm mu={mu_med:.2f} E={E_med:.0f}  "
          f"depth[{d_lo*1e3:.2f},{d_hi*1e3:.2f}]mm  |shear|[0,{s_hi*1e3:.2f}]mm")

    rep = {"context": {"R": R_med, "mu": mu_med, "E": E_med}, "sweeps": {}}

    def make_rows(depths, shear_mag):
        n = len(depths)
        rows = np.tile(base, (n, 1))
        rows[:, 2] = depths
        rows[:, 4] = shear_mag                      # sx = |shear| (sy=0), pure-x drag
        rows[:, 5] = 0.0
        return rows

    # ---- sweep 1: action = depth, fixed high shear ----
    depths = np.linspace(d_lo, d_hi, args.sweep_n)
    rows = make_rows(depths, s_hi)
    pf, pg = fno_proxy(rows), gt_proxy(rows)
    rep["sweeps"]["depth"] = {"depth_mm": (depths * 1e3).tolist(),
                              "fno_proxy": pf.tolist(), "gt_proxy": pg.tolist()}
    print("\n--- sweep DEPTH (action), shear fixed high ---")
    print(f"{'depth_mm':>9s} {'FNO_proxy':>11s} {'GT_proxy':>10s}")
    for d, a, g in zip(depths, pf, pg):
        print(f"{d*1e3:9.3f} {a:11.5f} {g:10.5f}")
    print(f"FNO depth-monotonic (proxy hi->lo as press harder?): "
          f"d(proxy)/d(depth) sign = {np.sign(pf[-1]-pf[0]):+.0f}  "
          f"range={pf.max()-pf.min():.5f} ({100*(pf.max()-pf.min())/(pf.mean()+1e-9):.0f}% of mean)")

    # ---- sweep 2: disturbance = |shear|, fixed mid depth ----
    shears = np.linspace(s_lo, s_hi, args.sweep_n)
    rows2 = np.tile(base, (args.sweep_n, 1)); rows2[:, 2] = base[2]
    rows2[:, 4] = shears; rows2[:, 5] = 0.0
    pf2, pg2 = fno_proxy(rows2), gt_proxy(rows2)
    rep["sweeps"]["shear"] = {"shear_mm": (shears * 1e3).tolist(),
                              "fno_proxy": pf2.tolist(), "gt_proxy": pg2.tolist()}
    print("\n--- sweep |SHEAR| (disturbance), depth fixed mid ---")
    print(f"{'shear_mm':>9s} {'FNO_proxy':>11s} {'GT_proxy':>10s}")
    for s, a, g in zip(shears, pf2, pg2):
        print(f"{s*1e3:9.3f} {a:11.5f} {g:10.5f}")
    print(f"FNO shear-monotonic (proxy up as more disturbance?): "
          f"sign = {np.sign(pf2[-1]-pf2[0]):+.0f}  range={pf2.max()-pf2.min():.5f}")

    # verdict
    depth_resp = abs(pf.max() - pf.min()) / (pf.mean() + 1e-9)
    shear_resp = abs(pf2.max() - pf2.min()) / (pf2.mean() + 1e-9)
    rep["verdict"] = {"depth_response_frac": float(depth_resp),
                      "shear_response_frac": float(shear_resp),
                      "depth_is_lever": bool(depth_resp > 0.05)}
    print(f"\nVERDICT: depth response = {depth_resp*100:.0f}% of mean, "
          f"shear response = {shear_resp*100:.0f}% of mean.")
    print("  depth is a usable control lever" if depth_resp > 0.05 else
          "  depth is WEAK -- pick a different action (e.g. counter-shear)")

    phase_dir = RUNS / "phase4"; ensure(phase_dir)
    json.dump(rep, open(phase_dir / "probe.json", "w"), indent=2)
    print(f"\nSaved {phase_dir / 'probe.json'}")


if __name__ == "__main__":
    main()
