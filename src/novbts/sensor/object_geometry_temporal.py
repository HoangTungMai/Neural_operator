#!/usr/bin/env python3
"""Phase 6d -- temporal marker-dot video over the loading path, PER object geometry.

Like sensor/temporal.py (which animated the 3 load modes), but the panels are now the
3 contact geometries (sphere / flat / cylinder), each driven through its own FEM loading
trajectory (disp_traj, requires --save-trajectory). Renders the differentiable marker-dot
sensor as an animated GIF (the geometries press/drag side by side), a static montage, and
a slip-signal curve (mean marker flow vs load fraction) so the geometry-dependent imprint
is visible as it develops, not just at the final frame.

  python -m novbts.sensor.object_geometry_temporal --geoms sphere flat cylinder
"""
import argparse
import json

import numpy as np
import torch

from novbts.operator.field2field import DEV
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, sample_field_to_markers,
    sensor_marker_grid_pixel_even, marker_half_extent,
)
from novbts.paths import FEM, RUNS, ensure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geoms", nargs="+", default=["sphere", "flat", "cylinder"])
    ap.add_argument("--data-root", default=str(FEM / "geom"))
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

    render_kw = dict(background=args.background, contrast=args.contrast, polarity="dark", saturate=True)
    geoms = []   # (name, traj[T,M,3], fracs, side)
    for g in args.geoms:
        z = np.load(f"{args.data_root}/{g}/fem_gt_shear.npz", allow_pickle=True)
        if "disp_traj" not in z.files:
            raise SystemExit(f"{g}: no disp_traj -- regenerate with --save-trajectory")
        coords, traj, fracs, params = z["coords"], z["disp_traj"], z["traj_fracs"], z["params"]
        smag = np.hypot(params[:, 4], params[:, 5])
        fi = int(np.argmax(smag))
        side = int(round(coords.shape[0] ** 0.5))
        geoms.append((g, traj[fi], fracs, side))
        print(f"[{g:9s}] frame={fi} T={traj.shape[1]} shear_mag={smag[fi]:.5f}m")
    coords0 = np.load(f"{args.data_root}/{args.geoms[0]}/fem_gt_shear.npz", allow_pickle=True)["coords"]
    T = geoms[0][1].shape[0]
    fracs = geoms[0][2]

    # shared sensor
    cam = PinholeCamera.from_gel(marker_half_extent(coords0), px=args.px, working_dist=args.working_dist)
    dense_t = torch.tensor(coords0, device=DEV)
    sensor_coords = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    sensor_t = torch.tensor(sensor_coords, device=DEV)
    m = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(sensor_t, torch.zeros(1, m, 3, device=DEV)))   # [1,m,2]

    def traj_to_pix(traj_TMC, side):
        fld = torch.tensor(traj_TMC, device=DEV).view(T, side, side, 3).permute(0, 3, 1, 2)
        mk = sample_field_to_markers(fld, dense_t, sensor_t)         # [T,m,3]
        return cam.project(deformed_marker_xyz(sensor_t, mk))        # [T,m,2]

    per = {g: traj_to_pix(tr, side) for g, tr, _, side in geoms}     # each [T,m,2]
    slip_curves = {g: (per[g] - pix_rest).norm(dim=-1).mean(1).cpu().numpy().tolist() for g, *_ in geoms}

    rep = {"data_root": args.data_root, "geoms": args.geoms, "T": int(T),
           "fracs": fracs.tolist(), "slip_curves": slip_curves}

    out_dir = RUNS / "phase6"; ensure(out_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        pr = pix_rest[0].cpu().numpy()
        im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        ng = len(geoms)

        def draw(ax, pij, title):
            img = render_dots(pij[None], args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
            fl = (pij - pix_rest[0]).cpu().numpy()
            ax.imshow(img, **im_kw)
            ax.quiver(pr[:, 0], pr[:, 1], fl[:, 0], -fl[:, 1], color="red",
                      scale_units="xy", angles="xy", scale=0.5, width=0.006)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

        # animated GIF: 3 geometry panels animating together (interpolated)
        Tn = (T - 1) * max(1, args.gif_interp) + 1
        frames = []
        for s in np.linspace(0, T - 1, Tn):
            i0 = int(np.floor(s)); i1 = min(i0 + 1, T - 1); a = float(s - i0); f = s / (T - 1)
            fig, axg = plt.subplots(1, ng, figsize=(3.2 * ng, 3.6), squeeze=False)
            for c, (g, *_ ) in enumerate(geoms):
                pij = (1 - a) * per[g][i0] + a * per[g][i1]
                draw(axg[0, c], pij, f"{g}   f={f:.2f}")
            fig.tight_layout(); fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frames.append(Image.fromarray(buf[..., :3].copy())); plt.close(fig)
        frames[0].save(out_dir / "object_geometry.gif", save_all=True,
                       append_images=frames[1:], duration=args.gif_ms, loop=0)
        rep["gif"] = str(out_dir / "object_geometry.gif")
        print(f"saved {out_dir/'object_geometry.gif'}  ({len(frames)} frames, {ng} geoms)")

        # static montage: rows=geom, cols=time snapshots
        ncol = min(T, 8); cols = np.linspace(0, T - 1, ncol).round().astype(int)
        fig2, axes = plt.subplots(ng, ncol, figsize=(2.1 * ncol, 2.3 * ng), squeeze=False)
        for r, (g, *_ ) in enumerate(geoms):
            for c, t in enumerate(cols):
                draw(axes[r, c], per[g][t], (f"{g}\n" if c == 0 else "") + f"f={fracs[t]:.2f}")
        fig2.suptitle("Marker-dot stream over loading path, per object geometry", fontsize=11)
        fig2.tight_layout(); fig2.savefig(out_dir / "object_geometry_video.png", dpi=130); plt.close(fig2)
        rep["montage"] = str(out_dir / "object_geometry_video.png")
        print(f"saved {out_dir/'object_geometry_video.png'}")

        # slip-signal curve
        fig3, ax = plt.subplots(figsize=(6.5, 4.2))
        for g, c in slip_curves.items():
            ax.plot(fracs, c, "-o", ms=4, label=g)
        ax.set_xlabel("load fraction f"); ax.set_ylabel("mean marker flow (px)")
        ax.set_title("Sensor slip signal vs loading (per object geometry)")
        ax.grid(alpha=0.3); ax.legend()
        fig3.tight_layout(); fig3.savefig(out_dir / "object_geometry_slip.png", dpi=130); plt.close(fig3)
        rep["slip_curve"] = str(out_dir / "object_geometry_slip.png")
        print(f"saved {out_dir/'object_geometry_slip.png'}")
    except Exception as e:
        rep["plot_error"] = str(e)
        print(f"plot skipped: {e}")

    json.dump(rep, open(out_dir / "object_geometry_temporal.json", "w"), indent=2, default=float)
    print(f"saved {out_dir/'object_geometry_temporal.json'}")


if __name__ == "__main__":
    main()
