#!/usr/bin/env python3
"""Phase 6c -- temporal sensor video: FEM ground truth vs FNO surrogate, side by side.

The FEM trajectory (disp_traj, from PhysX) is the GT loading sequence. The FNO predicts
the quasi-static field at each load fraction by feeding the lateral waypoint at that step
(path_xy(f) for the frame's load mode) as the shear input. Renders both through the
marker-dot sensor as a 2-panel animation (GT | FNO) and reports the per-step tracking error.

  python -m novbts.sensor.temporal_compare --data data/fem/traj_mix/fem_gt_shear.npz
"""
import argparse
import json

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, train_operator, params_to_fieldinput, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, sample_field_to_markers,
    sensor_marker_grid_pixel_even, marker_half_extent,
)
from novbts.paths import FEM, RUNS, ensure

LOAD_MODES = ["linear", "ortho", "reverse"]


def path_xy(f, sx, sy, mode):
    """mirror isaac_extract_shear.path_xy (no isaac import needed here)."""
    if mode == "ortho":
        return (sx * (f / 0.5), 0.0) if f <= 0.5 else (sx, sy * ((f - 0.5) / 0.5))
    if mode == "reverse":
        s = 1.5 * (f / 0.66) if f <= 0.66 else 1.5 - 0.5 * ((f - 0.66) / 0.34)
        return sx * s, sy * s
    return sx * f, sy * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fem/traj_mix/fem_gt_shear.npz")
    ap.add_argument("--fno-data", default=str(FEM / "shear_fine_swept_normaug.npz"),
                    help="dataset to TRAIN the FNO on (same gel geometry as --data)")
    ap.add_argument("--fno-epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-marker-side", type=int, default=11)
    ap.add_argument("--marker-pixel-fill", type=float, default=0.75)
    ap.add_argument("--working-dist", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    ap.add_argument("--gif-interp", type=int, default=4)
    ap.add_argument("--gif-ms", type=int, default=120)
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    if "disp_traj" not in z.files:
        raise SystemExit(f"{args.data} has no disp_traj -- regenerate with --save-trajectory")
    coords, traj, fracs = z["coords"], z["disp_traj"], z["traj_fracs"]
    params, mode, load_mode = z["params"], z["mode"], z["load_mode"]
    N, T, M, _ = traj.shape
    side = int(round(M ** 0.5))
    print(f"device={DEV}  temporal GT-vs-FNO  data={args.data}  N={N} T={T} side={side}")

    # ---- train the FNO (same recipe as the benchmark) ----
    Dt = load(args.fno_data)
    inp, out, scal, md = (Dt[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    ntr = inp.shape[0] - 400
    tr = torch.arange(0, ntr, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    secs, _ = train_operator(fno, (inp[tr] - im) / istd, (out[tr] - om) / ostd,
                             (scal[tr] - sm) / sstd, md[tr], cg, args.fno_epochs, 1e-3)
    fno.eval()
    print(f"[FNO] trained {secs:.0f}s on {args.fno_data} ({count_parameters(fno)} params)")

    # ---- sensor ----
    cam = PinholeCamera.from_gel(marker_half_extent(coords), px=args.px, working_dist=args.working_dist)
    dense_t = torch.tensor(coords, device=DEV)
    sensor_coords = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    sensor_t = torch.tensor(sensor_coords, device=DEV)
    m = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(sensor_t, torch.zeros(1, m, 3, device=DEV)))
    render_kw = dict(background=args.background, contrast=args.contrast, polarity="dark", saturate=True)

    def field_to_pix(disp_MC):
        """disp [M,3] (dense) -> sensor pixel positions [m,2]."""
        fld = torch.as_tensor(disp_MC, device=DEV, dtype=torch.float32).view(1, side, side, 3).permute(0, 3, 1, 2)
        mk = sample_field_to_markers(fld, dense_t, sensor_t)
        return cam.project(deformed_marker_xyz(sensor_t, mk))[0]

    @torch.no_grad()
    def fno_disp(params_row, sx, sy):
        pr = params_row.copy(); pr[4], pr[5] = sx, sy
        inp_np, scal_np = params_to_fieldinput(pr[None], coords, side)
        f = fno((torch.tensor(inp_np, device=DEV) - im) / istd,
                (torch.tensor(scal_np, device=DEV) - sm) / sstd) * ostd + om
        return f[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()   # [M,3]

    # one representative high-shear episode PER load mode
    smag = np.hypot(params[:, 4], params[:, 5])
    picks = []
    for lmid in range(len(LOAD_MODES)):
        idx = np.where((load_mode == lmid) & (mode >= 2))[0]
        if len(idx) == 0:
            idx = np.where(load_mode == lmid)[0]
        if len(idx):
            picks.append((lmid, int(idx[np.argmax(smag[idx])])))
    print("episodes:", {LOAD_MODES[l]: fi for l, fi in picks})

    # ---- per-snapshot tracking error (dense field GT vs FNO), per mode ----
    per_mode_rel = {}
    for lmid, fi in picks:
        sx_t, sy_t = float(params[fi, 4]), float(params[fi, 5]); lm = LOAD_MODES[lmid]
        rl = []
        for t in range(T):
            wx, wy = path_xy(float(fracs[t]), sx_t, sy_t, lm)
            rl.append(float(np.linalg.norm(fno_disp(params[fi], wx, wy) - traj[fi, t])
                            / (np.linalg.norm(traj[fi, t]) + 1e-9)))
        per_mode_rel[lm] = rl
        print(f"  {lm:7s} frame={fi} per-step rel-L2: " + " ".join(f"{r:.2f}" for r in rl))
    all_rel = [r for rl in per_mode_rel.values() for r in rl]

    rep = {"data": args.data, "fno_data": args.fno_data,
           "episodes": {LOAD_MODES[l]: fi for l, fi in picks},
           "fracs": fracs.tolist(), "per_mode_rel_l2": per_mode_rel,
           "mean_rel_l2": float(np.mean(all_rel))}

    phase_dir = RUNS / "phase6"; ensure(phase_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        pr_np = pix_rest[0].cpu().numpy()
        im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")

        def panel(ax, disp_MC, title):
            pix = field_to_pix(disp_MC)
            img = render_dots(pix[None], args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
            fl = (pix - pix_rest[0]).cpu().numpy()
            ax.imshow(img, **im_kw)
            ax.quiver(pr_np[:, 0], pr_np[:, 1], fl[:, 0], -fl[:, 1],
                      color="red", scale_units="xy", angles="xy", scale=0.5, width=0.005)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0); ax.set_xticks([]); ax.set_yticks([])

        nr = len(picks)
        Tn = (T - 1) * max(1, args.gif_interp) + 1
        frames = []
        for s in np.linspace(0, T - 1, Tn):
            i0 = int(np.floor(s)); i1 = min(i0 + 1, T - 1); a = float(s - i0); f = s / (T - 1)
            fig, axes = plt.subplots(nr, 2, figsize=(7.0, 3.5 * nr), squeeze=False)
            for r, (lmid, fi) in enumerate(picks):
                sx_t, sy_t = float(params[fi, 4]), float(params[fi, 5]); lm = LOAD_MODES[lmid]
                gt = (1 - a) * traj[fi, i0] + a * traj[fi, i1]
                wx, wy = path_xy(f, sx_t, sy_t, lm)
                panel(axes[r, 0], gt, f"FEM GT  [{lm}]  f={f:.2f}")
                panel(axes[r, 1], fno_disp(params[fi], wx, wy), f"FNO  [{lm}]  f={f:.2f}")
            fig.tight_layout(); fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frames.append(Image.fromarray(buf[..., :3].copy())); plt.close(fig)
        frames[0].save(phase_dir / "temporal_gt_vs_fno.gif", save_all=True,
                       append_images=frames[1:], duration=args.gif_ms, loop=0)
        rep["gif"] = str(phase_dir / "temporal_gt_vs_fno.gif")
        print(f"saved {phase_dir/'temporal_gt_vs_fno.gif'}  ({len(frames)} frames, {nr} modes x GT/FNO)")

        # static montage: per mode two rows (GT, FNO) across snapshots
        ncol = min(T, 8); cols = np.linspace(0, T - 1, ncol).round().astype(int)
        fig2, axes = plt.subplots(2 * nr, ncol, figsize=(2.1 * ncol, 2.3 * 2 * nr), squeeze=False)
        for r, (lmid, fi) in enumerate(picks):
            sx_t, sy_t = float(params[fi, 4]), float(params[fi, 5]); lm = LOAD_MODES[lmid]
            for c, t in enumerate(cols):
                wx, wy = path_xy(float(fracs[t]), sx_t, sy_t, lm)
                panel(axes[2 * r, c], traj[fi, t], (f"{lm} GT\n" if c == 0 else "") + f"f={fracs[t]:.2f}")
                panel(axes[2 * r + 1, c], fno_disp(params[fi], wx, wy), (f"{lm} FNO\n" if c == 0 else "") + f"f={fracs[t]:.2f}")
        fig2.suptitle(f"Temporal FEM GT vs FNO per load mode (mean rel-L2={np.mean(all_rel):.2f})", fontsize=11)
        fig2.tight_layout(); fig2.savefig(phase_dir / "temporal_gt_vs_fno.png", dpi=120); plt.close(fig2)
        rep["montage"] = str(phase_dir / "temporal_gt_vs_fno.png")
        print(f"saved {phase_dir/'temporal_gt_vs_fno.png'}")
    except Exception as e:
        rep["plot_error"] = str(e)
        print(f"plot skipped: {e}")

    json.dump(rep, open(phase_dir / "temporal_compare.json", "w"), indent=2, default=float)
    print(f"saved {phase_dir/'temporal_compare.json'}")


if __name__ == "__main__":
    main()
