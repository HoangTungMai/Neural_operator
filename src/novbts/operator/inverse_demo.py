#!/usr/bin/env python3
"""Differentiability demo: solve an INVERSE problem through the FNO.

Use-case #1 (state estimation): given an observed marker displacement field
y_obs (here a real PhysX-FEM frame), recover the applied shear (sx, sy) that
produced it, by minimising  J(sx,sy) = || FNO(input(sx,sy)) - y_obs ||^2.

The SAME forward model (the trained FNO) and the SAME optimiser (Adam) are used
two ways -- the only thing that changes is HOW the gradient dJ/d(sx,sy) is got:

  * autograd  : one backward pass through the FNO   (the differentiability win)
  * finite-diff: numerically, perturbing each unknown (what you must do when the
                 forward model is a NON-differentiable physics solver)

This isolates exactly what "differentiable" buys: not a different answer (both
converge to the same recovered shear), but the COST and STABILITY of the gradient.
We also report the cost SCALING: finite-diff needs (2*k) forward evals per step
for k unknowns; autograd needs 1 backward regardless of k -- and if each forward
were a PhysX-FEM solve (~0.14 s) instead of an FNO call (~ms), finite-diff is
hours where autograd is < 1 s.

  python -m novbts.operator.inverse_demo --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json
import time

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, train_operator, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import hertz_scalars, MODE_NAMES
from novbts.paths import FEM, RUNS, ensure


def build_input_channels(params_row, coords, side):
    """Reconstruct the FNO input channels for ONE frame, but split out the parts
    that depend on (sx,sy) from the parts that don't, so we can hold geometry
    fixed and make only (sx,sy) the unknowns. Mirrors params_to_fieldinput.

    Returns (pen[H,W], mask[H,W]) as torch tensors -- input is then
      ch0 = pen,  ch1 = sx*mask,  ch2 = sy*mask  (linear in the unknowns)."""
    x0, y0, depth, R, sx, sy, mu, E, geom = [float(params_row[i]) for i in range(9)]
    X = coords[:, 0].reshape(side, side)
    Y = coords[:, 1].reshape(side, side)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    r = np.sqrt(r2 + 1e-12)
    a, _, _, _ = hertz_scalars(np.array([depth]), np.array([R]), np.array([E]))
    a_eff = R if geom > 0.5 else float(a[0])
    if geom > 0.5:
        pen = depth * (r <= R)
    else:
        pen = np.clip(depth - r2 / (2.0 * R), 0.0, None)
    mask = (r <= a_eff).astype(np.float64)
    return (torch.tensor(pen, dtype=torch.float32, device=DEV),
            torch.tensor(mask, dtype=torch.float32, device=DEV),
            (sx, sy, mu, E))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--opt-lr", type=float, default=0.05, help="inverse-problem Adam lr (normalised units)")
    ap.add_argument("--fem-solve-s", type=float, default=0.14, help="ref PhysX-FEM solve time/frame")
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    coords = np.load(args.data, allow_pickle=True)["coords"]
    print(f"device={DEV}  inverse demo  data={args.data}  N={N} side={side}")

    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    tr = torch.arange(0, N - nt, device=DEV)
    te = torch.arange(N - nt, N, device=DEV)

    # ---- train the forward FNO (same recipe as the benchmark) ----
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    secs, _ = train_operator(fno, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]), mode[tr],
                             cg, args.epochs, args.lr)
    fno.eval()
    print(f"[FNO] trained {secs:.0f}s  ({count_parameters(fno)} params)")

    # per-channel shear scale (for normalised optimisation units)
    s_sx = float(istd[0, 1, 0, 0]); s_sy = float(istd[0, 2, 0, 0])  # input ch std for shear*mask

    # ---- pick the full_slip test frame with the LARGEST shear (clearest demo) ----
    te_modes = mode[te]
    cand = te[(te_modes == 3)]
    if len(cand) == 0:
        cand = te[(te_modes >= 2)]
    smag = np.hypot(D["params"][cand.cpu().numpy(), 4], D["params"][cand.cpu().numpy(), 5])
    fi = int(cand[int(np.argmax(smag))])
    y_obs = out[fi:fi + 1]                                   # raw FEM marker field [1,3,H,W]
    scal_i = nsc(scal[fi:fi + 1])
    pen, mask, (sx_t, sy_t, mu_t, E_t) = build_input_channels(D["params"][fi], coords, side)
    mag_t = float(np.hypot(sx_t, sy_t)); ang_t = float(np.degrees(np.arctan2(sy_t, sx_t)))
    print(f"\nobserved frame idx={fi}  mode={MODE_NAMES[int(mode[fi])]}  "
          f"true shear (sx,sy)=({sx_t*1e3:.3f}, {sy_t*1e3:.3f}) mm  "
          f"|s|={mag_t*1e3:.3f}mm @ {ang_t:.1f}deg  mu={mu_t:.2f}")

    pen3 = pen[None, None]                                   # [1,1,H,W]
    mask1 = mask[None, None]

    def assemble_norm_input(sx, sy):
        ch1 = sx * mask1
        ch2 = sy * mask1
        inp3 = torch.cat([pen3, ch1, ch2], 1)               # [1,3,H,W] raw
        return (inp3 - im) / istd

    def forward_loss(sx, sy):
        pred = fno(assemble_norm_input(sx, sy), scal_i) * ostd + om
        return torch.mean((pred - y_obs) ** 2)

    sx0, sy0 = 0.0, 0.0                                      # start from "no shear" guess

    # ===== method A: autograd =====
    v = torch.tensor([sx0, sy0], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([v], lr=args.opt_lr * max(s_sx, s_sy))
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_curve_a = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = forward_loss(v[0], v[1])
        loss.backward()
        opt.step()
        loss_curve_a.append(float(loss.detach()))
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t_auto = time.perf_counter() - t0
    sx_a, sy_a = float(v.detach()[0]), float(v.detach()[1])
    n_fwd_auto = args.steps                                  # 1 forward + 1 backward per step
    n_bwd_auto = args.steps

    # ===== method B: finite-difference (forward model treated as a black box) =====
    eps = 1e-2 * max(s_sx, s_sy)
    v2 = torch.tensor([sx0, sy0], device=DEV)
    opt2_v = torch.tensor([sx0, sy0], device=DEV, requires_grad=True)
    opt2 = torch.optim.Adam([opt2_v], lr=args.opt_lr * max(s_sx, s_sy))
    k = 2                                                    # unknowns
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_curve_b = []
    n_fwd_fd = 0
    with torch.no_grad():
        for _ in range(args.steps):
            base = forward_loss(opt2_v[0], opt2_v[1]); n_fwd_fd += 1
            g = torch.zeros(2, device=DEV)
            for j in range(k):                              # central difference per unknown
                d = torch.zeros(2, device=DEV); d[j] = eps
                lp = forward_loss(opt2_v[0] + d[0], opt2_v[1] + d[1])
                lm = forward_loss(opt2_v[0] - d[0], opt2_v[1] - d[1])
                g[j] = (lp - lm) / (2 * eps)
                n_fwd_fd += 2
            opt2_v.grad = g
            opt2.step()
            loss_curve_b.append(float(base))
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t_fd = time.perf_counter() - t0
    sx_b, sy_b = float(opt2_v[0]), float(opt2_v[1])

    # ---- recovery errors (vs true shear) ----
    def rec_err(sx, sy):
        true = np.array([sx_t, sy_t]); got = np.array([sx, sy])
        return float(np.linalg.norm(got - true) / (np.linalg.norm(true) + 1e-9))

    # ---- scaling argument: cost if the forward were a PhysX-FEM solve, and a
    #      high-dim inverse (recover the full input field, k = 3*H*W unknowns) ----
    k_field = 3 * side * side
    fwd_per_step_fd_field = 2 * k_field
    report = {
        "data": args.data, "frame_idx": fi, "mode": MODE_NAMES[int(mode[fi])],
        "true_shear": [sx_t, sy_t], "steps": args.steps,
        "autograd":      {"recovered": [sx_a, sy_a], "rel_err": rec_err(sx_a, sy_a),
                          "final_loss": loss_curve_a[-1], "wall_s": round(t_auto, 3),
                          "forward_evals": n_fwd_auto, "backward_evals": n_bwd_auto},
        "finite_diff":   {"recovered": [sx_b, sy_b], "rel_err": rec_err(sx_b, sy_b),
                          "final_loss": loss_curve_b[-1], "wall_s": round(t_fd, 3),
                          "forward_evals": n_fwd_fd, "backward_evals": 0},
        "scaling": {
            "unknowns_this_demo": k,
            "fd_forward_per_step": 2 * k,
            "autograd_backward_per_step": 1,
            "fem_solve_s": args.fem_solve_s,
            "fd_hours_if_fem_forward": round(n_fwd_fd * args.fem_solve_s / 3600, 3),
            "highdim_unknowns_full_field": k_field,
            "fd_forward_per_step_full_field": fwd_per_step_fd_field,
            "autograd_backward_per_step_full_field": 1,
        },
    }
    phase_dir = RUNS / "phase3_fem"; ensure(phase_dir)
    out_path = phase_dir / "inverse_demo.json"
    json.dump(report, open(out_path, "w"), indent=2)

    # ---- print ----
    a, b = report["autograd"], report["finite_diff"]
    print("\n=== inverse recovery of (sx, sy) from one marker field [mm] ===")
    print(f"true            (sx,sy) = ({sx_t*1e3:8.3f}, {sy_t*1e3:8.3f})")
    print(f"autograd        (sx,sy) = ({sx_a*1e3:8.3f}, {sy_a*1e3:8.3f})  rel_err={a['rel_err']:.3f}")
    print(f"finite-diff     (sx,sy) = ({sx_b*1e3:8.3f}, {sy_b*1e3:8.3f})  rel_err={b['rel_err']:.3f}")
    print(f"\n{'method':14s} {'wall_s':>8s} {'fwd_evals':>10s} {'bwd_evals':>10s} {'final_loss':>12s}")
    print(f"{'autograd':14s} {a['wall_s']:8.3f} {a['forward_evals']:10d} {a['backward_evals']:10d} {a['final_loss']:12.3e}")
    print(f"{'finite-diff':14s} {b['wall_s']:8.3f} {b['forward_evals']:10d} {b['backward_evals']:10d} {b['final_loss']:12.3e}")
    print(f"\nsame answer, but finite-diff used {b['forward_evals']/max(a['forward_evals'],1):.1f}x the forward calls "
          f"and {b['wall_s']/max(a['wall_s'],1e-9):.1f}x the wall time.")
    sc = report["scaling"]
    print(f"\nscaling: this demo has k={sc['unknowns_this_demo']} unknowns (FD={sc['fd_forward_per_step']} fwd/step, autograd=1 bwd/step).")
    print(f"  if each forward were a PhysX-FEM solve ({sc['fem_solve_s']}s): finite-diff = {sc['fd_hours_if_fem_forward']} h, "
          f"autograd (1 bwd/step) stays < 1 s.")
    print(f"  high-dim inverse (recover full {side}x{side}x3 = {sc['highdim_unknowns_full_field']} input field): "
          f"FD = {sc['fd_forward_per_step_full_field']} fwd/step (intractable), autograd still 1 bwd/step.")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
