#!/usr/bin/env python3
"""Phase 6c -- temporal marker-dot sensor stream from the FEM loading trajectory.

The generator (--save-trajectory) now stores the lateral loading PATH as T snapshots
per frame (disp_traj[N,T,M,3], f=0 normal -> f=1 final). This renders that path through
the differentiable marker-dot sensor -> a marker VIDEO (progressive stick->slip), plus a
per-step slip signal (mean marker flow vs load fraction). For load-mode 'ortho'/'reverse'
the path visibly turns / overshoots-and-returns, illustrating path dependence.

  python -m novbts.sensor.temporal --data data/fem/traj_mix/fem_gt_shear.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from novbts.operator.field2field import DEV
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, sample_field_to_markers,
    sensor_marker_grid_pixel_even, marker_half_extent,
)
from novbts.paths import RUNS, ensure

LOAD_MODES = ["linear", "ortho", "reverse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fem/traj_mix/fem_gt_shear.npz")
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-marker-side", type=int, default=11)
    ap.add_argument("--marker-pixel-fill", type=float, default=0.75)
    ap.add_argument("--working-dist", type=float, default=None,
                    help="camera-to-membrane distance in metres; default scales with marker footprint")
    ap.add_argument("--working-dist-ratio", type=float, default=1.1,
                    help="used when --working-dist is omitted: distance = ratio * marker half-extent")
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    ap.add_argument("--gif-interp", type=int, default=4, help="interpolated sub-steps between snapshots (smoother GIF)")
    ap.add_argument("--gif-ms", type=int, default=120, help="GIF frame duration (ms)")
    ap.add_argument("--out-dir", default="phase7", help="subdirectory under runs/")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    if "disp_traj" not in z.files:
        raise SystemExit(f"{args.data} has no disp_traj -- regenerate with --save-trajectory")
    coords, traj, fracs = z["coords"], z["disp_traj"], z["traj_fracs"]
    load_mode = z["load_mode"] if "load_mode" in z.files else np.zeros(traj.shape[0], int)
    params, mode = z["params"], z["mode"]
    N, T, M, _ = traj.shape
    side = int(round(M ** 0.5))
    print(f"device={DEV}  temporal  data={args.data}  N={N} T={T} M={M} side={side}")

    marker_half = marker_half_extent(coords)
    working_dist = args.working_dist if args.working_dist is not None else args.working_dist_ratio * marker_half
    cam = PinholeCamera.from_gel(marker_half, px=args.px, working_dist=working_dist)
    dense_t = torch.tensor(coords, device=DEV)
    sensor_coords = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    sensor_t = torch.tensor(sensor_coords, device=DEV)
    m = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(sensor_t, torch.zeros(1, m, 3, device=DEV)))   # [1,m,2]
    render_kw = dict(background=args.background, contrast=args.contrast, polarity="dark", saturate=True)

    def traj_to_markers(frame_idx):
        """[T,m,3] visible marker displacements over the loading path for one frame."""
        fld = torch.tensor(traj[frame_idx], device=DEV).view(T, side, side, 3).permute(0, 3, 1, 2)
        return sample_field_to_markers(fld, dense_t, sensor_t)      # [T,m,3]

    def markers_to_pix(markers):
        """[B,m,3] marker displacements -> [B,m,2] sensor pixel positions."""
        return cam.project(deformed_marker_xyz(sensor_t, markers))

    def traj_to_pix(frame_idx):
        """[T,m,2] sensor pixel positions over the loading path for one frame."""
        return markers_to_pix(traj_to_markers(frame_idx))

    # one representative high-shear frame per load mode
    smag = np.hypot(params[:, 4], params[:, 5])
    picks = []
    for lm in range(len(LOAD_MODES)):
        idx = np.where((load_mode == lm) & (mode >= 2))[0]
        if len(idx) == 0:
            idx = np.where(load_mode == lm)[0]
        if len(idx):
            picks.append((lm, int(idx[np.argmax(smag[idx])])))

    slip_curves = {}
    field_stats = {}
    for lm, fi in picks:
        mk = traj_to_markers(fi)
        pix = markers_to_pix(mk)
        flow = (pix - pix_rest).norm(dim=-1).mean(1).cpu().numpy()  # mean marker flow vs t
        slip_curves[LOAD_MODES[lm]] = flow.tolist()
        mk_final = mk[-1:]
        z_only = mk_final.clone()
        z_only[..., :2] = 0.0
        xy_only = mk_final.clone()
        xy_only[..., 2] = 0.0
        z_flow = (markers_to_pix(z_only) - pix_rest).norm(dim=-1).mean().item()
        xy_flow = (markers_to_pix(xy_only) - pix_rest).norm(dim=-1).mean().item()

        fld = traj[fi].reshape(T, side, side, 3)
        depth_mm = -fld[..., 2] * 1e3
        uxy_mm = fld[..., :2] * 1e3
        rr = np.hypot(coords[:, 0].reshape(side, side) - coords[:, 0].mean(),
                      coords[:, 1].reshape(side, side) - coords[:, 1].mean())
        center = rr < np.percentile(rr, 10)
        edge = rr > np.percentile(rr, 75)
        mean_uxy = uxy_mm.reshape(T, -1, 2).mean(1)
        residual = uxy_mm - mean_uxy[:, None, None, :]
        residual_mag = np.linalg.norm(residual.reshape(T, -1, 2), axis=-1)
        field_stats[LOAD_MODES[lm]] = {
            "frame": int(fi),
            "depth_param_mm": float(params[fi, 2] * 1e3),
            "radius_param_mm": float(params[fi, 3] * 1e3),
            "shear_param_mm": float(smag[fi] * 1e3),
            "by_frac": [
                {
                    "frac": float(fracs[t]),
                    "abs_depth_mm_p95": float(np.percentile(depth_mm[t], 95)),
                    "local_center_minus_edge_depth_mm": float(depth_mm[t][center].mean() - depth_mm[t][edge].mean()),
                    "residual_uxy_mean_mm": float(residual_mag[t].mean()),
                }
                for t in range(T)
            ],
            "final_abs_depth_mm_p95": float(np.percentile(depth_mm[-1], 95)),
            "final_local_center_minus_edge_depth_mm": float(depth_mm[-1][center].mean() - depth_mm[-1][edge].mean()),
            "final_mean_rigid_uxy_mm": [float(x) for x in mean_uxy[-1]],
            "final_residual_uxy_mean_mm": float(residual_mag[-1].mean()),
            "final_residual_uxy_max_mm": float(residual_mag[-1].max()),
            "final_mean_marker_flow_px": float(flow[-1]),
            "final_z_only_marker_flow_px": float(z_flow),
            "final_xy_only_marker_flow_px": float(xy_flow),
        }

    rep = {"data": args.data, "N": int(N), "T": int(T), "fracs": fracs.tolist(),
           "picks": {LOAD_MODES[lm]: fi for lm, fi in picks}, "slip_curves": slip_curves,
           "trajectory_semantics": "f=0 is the settled pure-normal pressed state; f=1 is the final shear endpoint",
           "camera": {**cam.as_dict(), "marker_half_extent_m": float(marker_half),
                      "working_dist_auto": args.working_dist is None,
                      "working_dist_ratio": float(args.working_dist_ratio)}}
    rep["field_stats"] = field_stats

    phase_dir = RUNS / args.out_dir; ensure(phase_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # video montage: rows = load modes, cols = time snapshots
        ncol = min(T, 8)
        cols = np.linspace(0, T - 1, ncol).round().astype(int)
        fig, axes = plt.subplots(len(picks), ncol, figsize=(2.1 * ncol, 2.3 * len(picks)), squeeze=False)
        pr = pix_rest[0].cpu().numpy()
        im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        for r, (lm, fi) in enumerate(picks):
            pix = traj_to_pix(fi)
            imgs = render_dots(pix, args.px, args.px, args.sigma, **render_kw).squeeze(1).cpu().numpy()
            fl = (pix - pix_rest).cpu().numpy()
            for c, t in enumerate(cols):
                ax = axes[r, c]
                ax.imshow(imgs[t], **im_kw)
                ax.quiver(pr[:, 0], pr[:, 1], fl[t][:, 0], -fl[t][:, 1],
                          color="red", scale_units="xy", angles="xy", scale=0.5, width=0.005)
                ax.set_title((f"{LOAD_MODES[lm]}\n" if c == 0 else "") + f"f={fracs[t]:.2f}", fontsize=8)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Temporal marker-dot stream over the loading path (per load mode)", fontsize=11)
        fig.tight_layout(); fig.savefig(phase_dir / "temporal_video.png", dpi=130); plt.close(fig)
        rep["video"] = str(phase_dir / "temporal_video.png")
        print(f"saved {phase_dir/'temporal_video.png'}")

        # The marker camera mixes in-plane flow and z-perspective magnification.
        # Save the physical normal field explicitly so the mm-scale sphere imprint
        # can be read without relying on camera sensitivity alone.
        xx = coords[:, 0].reshape(side, side) * 1e3
        yy = coords[:, 1].reshape(side, side) * 1e3
        all_depth = -traj[..., 2] * 1e3
        dvmin = max(0.0, float(np.percentile(all_depth, 1)))
        dvmax = float(np.percentile(all_depth, 99))
        figd, axd = plt.subplots(len(picks), ncol, figsize=(2.1 * ncol, 2.25 * len(picks)), squeeze=False)
        for r, (lm, fi) in enumerate(picks):
            for c, t in enumerate(cols):
                ax = axd[r, c]
                fld = traj[fi, t].reshape(side, side, 3)
                depth = -fld[..., 2] * 1e3
                imd = ax.imshow(depth, origin="lower", cmap="magma",
                                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                                vmin=dvmin, vmax=dvmax)
                u = fld[..., :2] * 1e3
                ures = u - u.reshape(-1, 2).mean(0)
                step = max(1, side // 8)
                ax.quiver(xx[::step, ::step], yy[::step, ::step],
                          ures[::step, ::step, 0], ures[::step, ::step, 1],
                          color="cyan", angles="xy", scale_units="xy",
                          scale=0.08, width=0.004)
                ax.set_title((f"{LOAD_MODES[lm]}\n" if c == 0 else "") + f"f={fracs[t]:.2f}", fontsize=8)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
        figd.suptitle("Normal indentation -uz (mm) + detrended tangential residual", fontsize=11)
        figd.subplots_adjust(right=0.93, top=0.88, wspace=0.05, hspace=0.18)
        cax = figd.add_axes([0.945, 0.16, 0.012, 0.66])
        cb = figd.colorbar(imd, cax=cax); cb.set_label("-uz (mm)")
        figd.savefig(phase_dir / "temporal_depth_residual.png", dpi=130); plt.close(figd)
        rep["depth_residual"] = str(phase_dir / "temporal_depth_residual.png")
        print(f"saved {phase_dir/'temporal_depth_residual.png'}")

        if picks:
            lm, fi = picks[-1]
            fld = traj[fi, -1].reshape(side, side, 3)
            depth = -fld[..., 2] * 1e3
            rr = np.hypot(xx - xx.mean(), yy - yy.mean())
            edge = rr > np.percentile(rr, 75)
            local_depth = depth - depth[edge].mean()
            u = fld[..., :2] * 1e3
            umean = u.reshape(-1, 2).mean(0)
            ures = u - umean
            step = max(1, side // 10)
            figq, axq = plt.subplots(1, 4, figsize=(13.5, 3.6))
            im0 = axq[0].imshow(depth, origin="lower", cmap="magma",
                                extent=[xx.min(), xx.max(), yy.min(), yy.max()])
            axq[0].set_title("absolute -uz (mm)")
            im1 = axq[1].imshow(local_depth, origin="lower", cmap="coolwarm",
                                extent=[xx.min(), xx.max(), yy.min(), yy.max()])
            axq[1].set_title("local -uz minus edge (mm)")
            axq[2].imshow(depth, origin="lower", cmap="Greys",
                          extent=[xx.min(), xx.max(), yy.min(), yy.max()], alpha=0.35)
            axq[2].quiver(xx[::step, ::step], yy[::step, ::step],
                          u[::step, ::step, 0], u[::step, ::step, 1],
                          color="red", angles="xy", scale_units="xy", scale=0.7, width=0.004)
            axq[2].set_title(f"absolute uxy, mean=({umean[0]:.3f},{umean[1]:.3f}) mm")
            axq[3].imshow(local_depth, origin="lower", cmap="Greys",
                          extent=[xx.min(), xx.max(), yy.min(), yy.max()], alpha=0.35)
            axq[3].quiver(xx[::step, ::step], yy[::step, ::step],
                          ures[::step, ::step, 0], ures[::step, ::step, 1],
                          color="cyan", angles="xy", scale_units="xy", scale=0.08, width=0.004)
            axq[3].set_title("detrended uxy residual")
            for ax in axq:
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("x mm")
                ax.set_yticks([])
            figq.colorbar(im0, ax=axq[0], fraction=0.046, pad=0.02)
            figq.colorbar(im1, ax=axq[1], fraction=0.046, pad=0.02)
            figq.suptitle(f"Phase 7 final field diagnostic ({LOAD_MODES[lm]})", fontsize=11)
            figq.tight_layout()
            figq.savefig(phase_dir / "phase7_field_diagnostic.png", dpi=130); plt.close(figq)
            rep["field_diagnostic"] = str(phase_dir / "phase7_field_diagnostic.png")
            print(f"saved {phase_dir/'phase7_field_diagnostic.png'}")

        # slip signal vs load fraction
        fig2, ax = plt.subplots(figsize=(6.5, 4.2))
        for name, c in slip_curves.items():
            ax.plot(fracs, c, "-o", ms=4, label=name)
        ax.set_xlabel("load fraction f"); ax.set_ylabel("mean marker flow (px)")
        ax.set_title("Sensor slip signal vs loading (path-dependent)")
        ax.grid(alpha=0.3); ax.legend()
        fig2.tight_layout(); fig2.savefig(phase_dir / "temporal_slip_curve.png", dpi=130); plt.close(fig2)
        rep["slip_curve"] = str(phase_dir / "temporal_slip_curve.png")
        print(f"saved {phase_dir/'temporal_slip_curve.png'}")

        # animated GIF: 3 panels (load modes) animating together, interpolated for smoothness
        from PIL import Image
        per = {lm: traj_to_pix(fi) for lm, fi in picks}            # each [T,m,2] tensor on DEV
        Tn = (T - 1) * max(1, args.gif_interp) + 1
        gif_frames = []
        for s in np.linspace(0, T - 1, Tn):
            i0 = int(np.floor(s)); i1 = min(i0 + 1, T - 1); a = float(s - i0)
            figg, axg = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.5), squeeze=False)
            for c, (lm, fi) in enumerate(picks):
                pij = (1 - a) * per[lm][i0] + a * per[lm][i1]       # [m,2]
                img = render_dots(pij[None], args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
                fl = (pij - pix_rest[0]).cpu().numpy()
                ax = axg[0, c]
                ax.imshow(img, **im_kw)
                ax.quiver(pr[:, 0], pr[:, 1], fl[:, 0], -fl[:, 1],
                          color="red", scale_units="xy", angles="xy", scale=0.5, width=0.005)
                ax.set_title(f"{LOAD_MODES[lm]}   f={s/(T-1):.2f}", fontsize=10)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0); ax.set_xticks([]); ax.set_yticks([])
            figg.tight_layout(); figg.canvas.draw()
            w, h = figg.canvas.get_width_height()
            buf = np.frombuffer(figg.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            gif_frames.append(Image.fromarray(buf[..., :3].copy()))
            plt.close(figg)
        gif_frames[0].save(phase_dir / "temporal.gif", save_all=True,
                           append_images=gif_frames[1:], duration=args.gif_ms, loop=0)
        rep["gif"] = str(phase_dir / "temporal.gif")
        print(f"saved {phase_dir/'temporal.gif'}  ({len(gif_frames)} frames)")
    except Exception as e:
        rep["plot_error"] = str(e)
        print(f"plot skipped: {e}")

    json.dump(rep, open(phase_dir / "temporal.json", "w"), indent=2, default=float)
    print(f"saved {phase_dir/'temporal.json'}")


if __name__ == "__main__":
    main()
