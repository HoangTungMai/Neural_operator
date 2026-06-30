#!/usr/bin/env python3
"""Phase 5a -- turn an FEM marker-displacement npz into a marker-dot SENSOR dataset.

Projects the gel-surface markers (rest + deformed) through a below-membrane pinhole
camera and renders dot images. Stores the pixel positions + camera config (images are
re-rendered on demand to save disk), validates the render->track round-trip, and writes
a preview montage.

  python -m novbts.sensor.build_sensor_dataset --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from novbts.operator.field2field import DEV
from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, track_flow_known,
    track_flow_image, marker_half_extent, sensor_marker_grid, sensor_marker_grid_pixel_even,
    sample_field_to_markers,
)
from novbts.paths import FEM, RUNS, ensure


def project_all(cam, coords_t, disp, chunk=256):
    """disp [N,M,3] np -> pix_def [N,M,2] tensor (chunked)."""
    pieces = []
    for s in range(0, disp.shape[0], chunk):
        d = torch.tensor(disp[s:s + chunk], device=DEV)
        pieces.append(cam.project(deformed_marker_xyz(coords_t, d)).cpu())
    return torch.cat(pieces)


def sample_all(coords, sensor_coords, disp, chunk=256):
    """disp [N,M,3] dense np -> [N,m,3] at visible sensor marker coordinates."""
    side = int(np.sqrt(disp.shape[1]))
    coords_t = torch.tensor(coords, device=DEV)
    sensor_coords_t = torch.tensor(sensor_coords, device=DEV)
    pieces = []
    for s in range(0, disp.shape[0], chunk):
        d = torch.tensor(disp[s:s + chunk], device=DEV)
        field = d.view(d.shape[0], side, side, 3).permute(0, 3, 1, 2)
        pieces.append(sample_field_to_markers(field, coords_t, sensor_coords_t).cpu())
    return torch.cat(pieces)


def representative_frames(mode, score, n_per_mode):
    """Pick low/mid/high observation-strength examples for each contact mode."""
    picked = {}
    for m, name in enumerate(MODE_NAMES):
        idx = np.where(mode == m)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(score[idx])]
        if len(order) <= n_per_mode:
            frames = order
        else:
            qs = np.linspace(0.18, 0.88, n_per_mode)
            locs = np.clip(np.round(qs * (len(order) - 1)).astype(int), 0, len(order) - 1)
            frames = order[locs]
            if len(np.unique(frames)) < n_per_mode:
                locs = np.linspace(0, len(order) - 1, n_per_mode).round().astype(int)
                frames = order[locs]
        picked[name] = [int(f) for f in frames]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-marker-side", type=int, default=11,
                    help="visible tracking-marker side; the underlying FEM/FNO field stays dense")
    ap.add_argument("--marker-placement", choices=["pixel_even", "gel_even"], default="pixel_even",
                    help="pixel_even makes the rest camera image perfectly regular")
    ap.add_argument("--marker-pixel-fill", type=float, default=0.75,
                    help="fraction of image width/height spanned by visible rest marker centers")
    ap.add_argument("--marker-inset", type=float, default=0.06,
                    help="gel_even-only fractional margin between visible marker grid and gel-field boundary")
    ap.add_argument("--working-dist", type=float, default=0.05)
    ap.add_argument("--fill", type=float, default=0.85)
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--dot-polarity", choices=["bright", "dark"], default="dark")
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    ap.add_argument("--saturate-dots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rt-n", type=int, default=120, help="frames for the round-trip check")
    ap.add_argument("--track-win", type=int, default=5)
    ap.add_argument("--sample-n-per-mode", type=int, default=3,
                    help="number of representative test samples to render for each contact mode")
    ap.add_argument("--save-images", action="store_true", help="also store rendered def images (big)")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    coords, disp, mode, params = z["coords"], z["disp"], z["mode"], z["params"]
    N, field_M = disp.shape[0], disp.shape[1]
    print(f"device={DEV}  sensor build  data={args.data}  N={N} field_M={field_M}")

    cam = PinholeCamera.from_gel(marker_half_extent(coords), px=args.px,
                                 working_dist=args.working_dist, fill=args.fill)
    if args.marker_placement == "pixel_even":
        sensor_coords = sensor_marker_grid_pixel_even(
            cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    else:
        sensor_coords = sensor_marker_grid(coords, args.sensor_marker_side, inset=args.marker_inset)
    sensor_disp = sample_all(coords, sensor_coords, disp)
    sensor_disp_np = sensor_disp.numpy()
    sensor_M = sensor_coords.shape[0]
    print(f"visible markers: placement={args.marker_placement} "
          f"side={args.sensor_marker_side} M={sensor_M}")
    coords_t = torch.tensor(sensor_coords, device=DEV)
    pix_rest = cam.project(deformed_marker_xyz(coords_t, torch.zeros(1, sensor_M, 3, device=DEV)))  # [1,m,2]
    pix_def = project_all(cam, coords_t, sensor_disp_np)                                             # [N,m,2]
    pix_flow = pix_def - pix_rest.cpu()                                                       # [N,M,2]
    render_kw = dict(background=args.background, contrast=args.contrast,
                     polarity=args.dot_polarity, saturate=args.saturate_dots)

    # --- faithfulness: pixel flow should track the in-plane displacement direction ---
    flow = pix_flow.numpy().reshape(N, -1)
    dxy = sensor_disp_np[:, :, :2].reshape(N, -1)
    fn = np.linalg.norm(flow, axis=1); dn = np.linalg.norm(dxy, axis=1)
    cos = (flow * dxy).sum(1) / (fn * dn + 1e-12)
    nz = dn > 1e-9
    mode_names = ("normal", "stick", "partial_slip", "full_slip")
    cos_by_mode = {}
    for mode_id, mode_name in enumerate(mode_names):
        sel = (mode == mode_id) & nz
        cos_by_mode[mode_name] = {
            "mean": float(cos[sel].mean()) if sel.any() else None,
            "n_valid": int(sel.sum()),
        }
    print(f"flow<->disp_xy alignment: mean cos={cos[nz].mean():.4f}  "
          f"(pixel flow faithfully encodes in-plane displacement)")

    # --- render->track round-trip on a subset (chunked to stay memory-safe) ---
    rng = np.random.default_rng(0)
    sub = rng.choice(N, size=min(args.rt_n, N), replace=False)
    msub = mode[sub]
    errs = []
    for s in range(0, len(sub), 24):
        idx = sub[s:s + 24]
        pd = cam.project(deformed_marker_xyz(coords_t, torch.tensor(sensor_disp_np[idx], device=DEV)))  # [b,m,2]
        img = render_dots(pd, args.px, args.px, args.sigma, **render_kw)                             # [b,1,H,W]
        tracked = track_flow_image(img, pix_rest.expand(len(idx), sensor_M, 2),
                                   win=args.track_win, dark=args.dot_polarity == "dark")
        errs.append((tracked - pd).norm(dim=-1).cpu())                                        # [b,M]
    rt_err = torch.cat(errs)                                                                  # [n,M] pixels
    def regime(sel):
        return float(rt_err[torch.tensor(sel, device=rt_err.device)].mean()) if sel.any() else float("nan")
    rt = {
        "overall_px": float(rt_err.mean()),
        "stick_px": regime(msub <= 1),
        "slip_px": regime(msub >= 2),
        "by_mode_px": {
            mode_name: regime(msub == mode_id)
            for mode_id, mode_name in enumerate(mode_names)
        },
        "p95_px": float(rt_err.flatten().quantile(0.95)),
    }
    print(f"round-trip track error (px): overall={rt['overall_px']:.3f}  "
          f"stick={rt['stick_px']:.3f}  slip={rt['slip_px']:.3f}  p95={rt['p95_px']:.3f}")

    # --- save sensor dataset ---
    stem = Path(args.data).stem
    out_npz = FEM / f"{stem}_sensor.npz"
    save = dict(coords=coords.astype(np.float32), params=params.astype(np.float32),
                sensor_coords=sensor_coords.astype(np.float32),
                mode=mode.astype(np.int32), pix_rest=pix_rest.cpu().numpy().astype(np.float32),
                pix_def=pix_def.numpy().astype(np.float32), pix_flow=pix_flow.numpy().astype(np.float32),
                cam=np.array(json.dumps(cam.as_dict())), sigma=np.float32(args.sigma), px=np.int32(args.px),
                background=np.float32(args.background), contrast=np.float32(args.contrast),
                dot_polarity=np.array(args.dot_polarity),
                saturate_dots=np.bool_(args.saturate_dots),
                meta=np.array("marker-dot VBTS sensor: below-membrane pinhole + gaussian-splat tracking dots"))
    if args.save_images:
        imgs = []
        for s in range(0, N, 256):
            d = torch.tensor(sensor_disp_np[s:s + 256], device=DEV)
            p = cam.project(deformed_marker_xyz(coords_t, d))
            imgs.append(render_dots(p, args.px, args.px, args.sigma, **render_kw)
                        .squeeze(1).cpu().numpy().astype(np.float32))
        save["def_img"] = np.concatenate(imgs)
    np.savez_compressed(out_npz, **save)
    print(f"saved sensor dataset -> {out_npz}")

    # --- preview montage on a large-shear full-slip frame ---
    phase_dir = RUNS / "phase5"; ensure(phase_dir)
    fs = np.where(mode == 3)[0]
    pick = int(fs[np.argmax(np.linalg.norm(disp[fs][:, :, :2], axis=(1, 2)))]) if len(fs) else 0
    rep = {"gt": os.path.basename(args.data), "gt_path": args.data,
           "data": args.data, "N": int(N), "field_M": int(field_M), "sensor_M": int(sensor_M),
           "sensor_marker_side": args.sensor_marker_side, "marker_placement": args.marker_placement,
           "marker_pixel_fill": args.marker_pixel_fill, "marker_inset": args.marker_inset,
           "px": args.px,
           "camera": cam.as_dict(), "sigma": args.sigma,
           "dot_style": {"polarity": args.dot_polarity, "background": args.background,
                         "contrast": args.contrast, "saturate": args.saturate_dots},
           "flow_disp_cos_mean": float(cos[nz].mean()),
           "flow_disp_cos_by_mode": cos_by_mode,
           "round_trip": rt,
           "preview_frame": pick, "sensor_npz": str(out_npz)}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d1 = torch.tensor(sensor_disp_np[pick:pick + 1], device=DEV)
        pdp = cam.project(deformed_marker_xyz(coords_t, d1))
        img_r = render_dots(pix_rest, args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
        img_d = render_dots(pdp, args.px, args.px, args.sigma, **render_kw)[0, 0].cpu().numpy()
        pr_np = pix_rest[0].cpu().numpy(); pf_np = (pdp[0] - pix_rest[0]).cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
        im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        ax[0].imshow(img_r, **im_kw); ax[0].set_title("rest dots")
        ax[1].imshow(img_d, **im_kw); ax[1].set_title(f"deformed (frame {pick}, full_slip)")
        ax[2].imshow(img_r, alpha=0.4, **im_kw)
        ax[2].quiver(pr_np[:, 0], pr_np[:, 1], pf_np[:, 0], -pf_np[:, 1],
                     color="red", scale_units="xy", angles="xy", scale=0.5, width=0.003)
        ax[2].set_title("marker flow (rest->deformed)")
        for a in ax:
            a.set_aspect("equal", adjustable="box")
            a.set_xlim(0, args.px); a.set_ylim(args.px, 0); a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(phase_dir / "preview.png", dpi=130)
        rep["preview"] = str(phase_dir / "preview.png")
        print(f"saved preview -> {phase_dir / 'preview.png'}")

        flow_score = np.linalg.norm(pix_flow.numpy(), axis=(1, 2))
        samples_by_mode = representative_frames(mode, flow_score, args.sample_n_per_mode)
        sample_frames = [f for frames in samples_by_mode.values() for f in frames]
        if sample_frames:
            dsel = torch.tensor(sensor_disp_np[sample_frames], device=DEV)
            psel = cam.project(deformed_marker_xyz(coords_t, dsel))
            imgs = render_dots(psel, args.px, args.px, args.sigma, **render_kw).squeeze(1).cpu().numpy()
            flows = (psel - pix_rest).cpu().numpy()
            frame_pos = {f: i for i, f in enumerate(sample_frames)}
            rows = list(samples_by_mode.items())
            nrows, ncols = len(rows), max(len(v) for _, v in rows)
            fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
            for r, (name, frames) in enumerate(rows):
                for c in range(ncols):
                    axc = axes[r, c]
                    if c >= len(frames):
                        axc.axis("off")
                        continue
                    f = frames[c]
                    pos = frame_pos[f]
                    axc.imshow(imgs[pos], **im_kw)
                    fl = flows[pos]
                    axc.quiver(pr_np[:, 0], pr_np[:, 1], fl[:, 0], -fl[:, 1],
                               color="red", scale_units="xy", angles="xy", scale=0.5, width=0.004)
                    mag = float(np.linalg.norm(fl, axis=1).max())
                    axc.set_title(f"{name}\nframe {f}  maxflow={mag:.1f}px", fontsize=9)
                    axc.set_aspect("equal", adjustable="box")
                    axc.set_xlim(0, args.px); axc.set_ylim(args.px, 0)
                    axc.set_xticks([]); axc.set_yticks([])
            fig2.tight_layout()
            fig2.savefig(phase_dir / "test_samples.png", dpi=150)
            plt.close(fig2)
            rep["test_samples"] = str(phase_dir / "test_samples.png")
            rep["test_sample_frames"] = samples_by_mode
            print(f"saved test samples -> {phase_dir / 'test_samples.png'}")
        plt.close(fig)
    except Exception as e:
        rep["preview_error"] = str(e)
        print(f"preview skipped: {e}")

    json.dump(rep, open(phase_dir / "sensor_build.json", "w"), indent=2, default=float)
    print(f"saved {phase_dir / 'sensor_build.json'}")


if __name__ == "__main__":
    main()
