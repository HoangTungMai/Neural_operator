#!/usr/bin/env python3
"""Phase 6d -- object-geometry coverage: how the contact SHAPE changes the tactile imprint.

The generator now drives more than a sphere: --indentor-geom {sphere,flat,cylinder,mesh}.
Each produces a different gel-surface displacement field. This loads the per-geometry FEM
sweeps (data/fem/geom/<geom>/), renders a representative high-shear frame through the SAME
marker-dot sensor, and quantifies how geometry reshapes the imprint:

  - normal  : peak penetration uz + contact-area footprint (markers in contact)
  - tangential: mean marker flow (px) -- a square punch grips a broad area, a sphere a point

A 2-row montage (marker-dot image + flow quiver; uz depth map) per geometry shows the
qualitative difference; the table gives the numbers. Confirms the framework generalises
beyond the sphere -- a usable multi-object differentiable VBTS GT.

  python -m novbts.sensor.object_geometry --geoms sphere flat cylinder
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
    args = ap.parse_args()

    render_kw = dict(background=args.background, contrast=args.contrast, polarity="dark", saturate=True)
    rows = []   # (geom, coords, disp_frame, stats)
    for g in args.geoms:
        path = f"{args.data_root}/{g}/fem_gt_shear.npz"
        z = np.load(path, allow_pickle=True)
        coords, disp, params = z["coords"], z["disp"], z["params"]
        smag = np.hypot(params[:, 4], params[:, 5])
        fi = int(np.argmax(smag))                       # representative high-shear frame
        d = disp[fi]                                    # [M,3]
        side = int(round(coords.shape[0] ** 0.5))
        uz = d[:, 2]
        tang = np.linalg.norm(d[:, :2], axis=1)
        peak_uz = float(uz.min())
        contact = uz < 0.1 * peak_uz                    # markers meaningfully pressed in
        stats = {"frame": fi, "peak_uz_m": peak_uz, "mean_tang_m": float(tang.mean()),
                 "max_tang_m": float(tang.max()),
                 "contact_area_frac": float(contact.mean()),
                 "shear_mag_m": float(smag[fi])}
        rows.append((g, coords, d, side, stats))
        print(f"[{g:9s}] frame={fi}  peak_uz={peak_uz:.5f}m  mean_tang={tang.mean():.5f}m  "
              f"max_tang={tang.max():.5f}m  contact_area={contact.mean()*100:.0f}%")

    # shared sensor (geometry-independent gel footprint from the first dataset)
    coords0 = rows[0][1]
    cam = PinholeCamera.from_gel(marker_half_extent(coords0), px=args.px, working_dist=args.working_dist)
    dense_t = torch.tensor(coords0, device=DEV)
    sensor_coords = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    sensor_t = torch.tensor(sensor_coords, device=DEV)
    m = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(sensor_t, torch.zeros(1, m, 3, device=DEV)))[0]

    def field_to_pix(disp_MC, side):
        fld = torch.as_tensor(disp_MC, device=DEV, dtype=torch.float32).view(1, side, side, 3).permute(0, 3, 1, 2)
        mk = sample_field_to_markers(fld, dense_t, sensor_t)
        return cam.project(deformed_marker_xyz(sensor_t, mk))[0]

    rep = {"data_root": args.data_root, "geoms": {g: s for g, _, _, _, s in rows}}
    out_dir = RUNS / "phase6"; ensure(out_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ng = len(rows)
        fig, axes = plt.subplots(2, ng, figsize=(3.4 * ng, 6.6), squeeze=False)
        pr = pix_rest.cpu().numpy()
        for c, (g, coords, d, side, st) in enumerate(rows):
            pix = field_to_pix(d, side)
            img = render_dots(pix[None], args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
            fl = (pix - pix_rest).cpu().numpy()
            ax = axes[0, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.quiver(pr[:, 0], pr[:, 1], fl[:, 0], -fl[:, 1], color="red",
                      scale_units="xy", angles="xy", scale=0.5, width=0.006)
            ax.set_title(f"{g}\nmean flow {np.linalg.norm(fl,axis=1).mean():.2f}px", fontsize=10)
            ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            # uz depth map
            ax2 = axes[1, c]
            uzg = d[:, 2].reshape(side, side)
            im = ax2.imshow(uzg * 1e3, cmap="viridis_r", interpolation="bilinear")
            ax2.set_title(f"uz (mm)  peak {st['peak_uz_m']*1e3:.2f}  contact {st['contact_area_frac']*100:.0f}%",
                          fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig.suptitle("Object geometry shapes the tactile imprint (high-shear frame, same sensor)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / "object_geometry.png", dpi=130); plt.close(fig)
        rep["montage"] = str(out_dir / "object_geometry.png")
        print(f"saved {out_dir/'object_geometry.png'}")
    except Exception as e:
        rep["plot_error"] = str(e)
        print(f"plot skipped: {e}")

    json.dump(rep, open(out_dir / "object_geometry.json", "w"), indent=2, default=float)
    print(f"saved {out_dir/'object_geometry.json'}")


if __name__ == "__main__":
    main()
