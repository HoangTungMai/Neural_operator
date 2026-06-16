#!/usr/bin/env python3
"""Phase 6b -- sensor REALISM: camera noise model + real blob-tracker -> the noise FLOOR.

The GT-vs-FNO sensor metric (temporal_compare) reported EPE=0.38px, but with a NOISELESS
render the tracking floor was ~0, so "is 0.38px good?" had no reference. This module adds a
physically-plausible camera pipeline -- Poisson shot noise + Gaussian read noise + optical
blur + 8-bit quantisation -- and localises each marker with the REAL blob-tracker (intensity
centroid), the same readout a physical marker-dot VBTS runs. That gives a synthetic-but-honest
*tracking noise floor*: how far the tracked marker wanders from its true position under noise.

Verdict logic: if the FNO model error (0.38px) sits BELOW the noise floor at a realistic
camera noise level, the surrogate is indistinguishable from the sensor's own readout jitter --
i.e. good enough that sensor noise, not the FNO, is the limiting factor. No hardware needed;
the physical calibration that ties this to a real build is the deferred half (calibration.py).

  python -m novbts.sensor.realism --data data/fem/traj_mix/fem_gt_shear.npz
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from novbts.operator.field2field import (
    FNOField, train_operator, params_to_fieldinput, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, sample_field_to_markers,
    sensor_marker_grid_pixel_even, marker_half_extent, track_flow_image,
)
from novbts.paths import FEM, RUNS, ensure


def gaussian_blur(img, sigma):
    """Optical blur: separable Gaussian conv on [B,1,H,W]. sigma in px."""
    if sigma <= 0:
        return img
    rad = max(1, int(round(3 * sigma)))
    x = torch.arange(-rad, rad + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma)); k = k / k.sum()
    k2 = (k[:, None] * k[None, :])[None, None]
    return F.conv2d(img, k2, padding=rad)


def add_camera_noise(img, photons=300.0, read_noise=0.02, blur=0.0, quantize=True):
    """Realistic camera readout on a normalised intensity image [B,1,H,W] in ~[0,1].

      shot noise : Poisson(img*photons)/photons   (signal-dependent)
      read noise : + N(0, read_noise)             (additive, RMS as fraction of full-scale)
      blur       : optical PSF (Gaussian, px)
      quantize   : 8-bit ADC
    Seed torch.manual_seed(...) beforehand for reproducibility (torch.poisson uses global RNG).
    """
    x = gaussian_blur(img, blur) if blur > 0 else img
    x = torch.clamp(x, min=0.0)
    if photons and photons > 0:
        x = torch.poisson(x * photons) / photons
    if read_noise and read_noise > 0:
        x = x + read_noise * torch.randn_like(x)
    x = torch.clamp(x, 0.0, 1.0)
    if quantize:
        x = torch.round(x * 255.0) / 255.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fem/traj_mix/fem_gt_shear.npz")
    ap.add_argument("--fno-data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--fno-epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--n-frames", type=int, default=60, help="random test frames sampled for EPE stats")
    ap.add_argument("--noise-seeds", type=int, default=6, help="noise realisations averaged per frame")
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-marker-side", type=int, default=11)
    ap.add_argument("--marker-pixel-fill", type=float, default=0.75)
    ap.add_argument("--working-dist", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    ap.add_argument("--photons", type=float, default=300.0)
    ap.add_argument("--blur", type=float, default=0.6, help="optical PSF sigma (px) at the realistic operating point")
    ap.add_argument("--read-noise-op", type=float, default=0.02, help="realistic read-noise RMS (fraction of full scale)")
    ap.add_argument("--track-win", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    coords, params, mode = z["coords"], z["params"], z["mode"]
    disp = z["disp"] if "disp" in z.files else z["out"]
    N = disp.shape[0]
    side = int(round(coords.shape[0] ** 0.5))
    print(f"device={DEV}  realism/noise-floor  data={args.data}  N={N} side={side}")

    # ---- train FNO (same recipe as temporal_compare) ----
    Dt = load(args.fno_data)
    inp, out, scal, md = (Dt[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    ntr = inp.shape[0] - 400
    tr = torch.arange(0, ntr, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    secs, _ = train_operator(fno, (inp[tr] - im) / istd, (out[tr] - om) / ostd,
                             (scal[tr] - sm) / sstd, md[tr], None, args.fno_epochs, 1e-3)
    fno.eval()
    print(f"[FNO] trained {secs:.0f}s ({count_parameters(fno)} params)")

    # ---- sensor ----
    cam = PinholeCamera.from_gel(marker_half_extent(coords), px=args.px, working_dist=args.working_dist)
    dense_t = torch.tensor(coords, device=DEV)
    sensor_coords = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    sensor_t = torch.tensor(sensor_coords, device=DEV)
    m = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(sensor_t, torch.zeros(1, m, 3, device=DEV)))   # [1,m,2]
    render_kw = dict(background=args.background, contrast=args.contrast, polarity="dark", saturate=True)
    dmat = torch.cdist(pix_rest[0], pix_rest[0]).fill_diagonal_(float("inf"))
    pitch_px = float(dmat.min(dim=1).values.median())

    def field_to_pix(disp_MC):
        fld = torch.as_tensor(disp_MC, device=DEV, dtype=torch.float32).view(1, side, side, 3).permute(0, 3, 1, 2)
        mk = sample_field_to_markers(fld, dense_t, sensor_t)
        return cam.project(deformed_marker_xyz(sensor_t, mk))[0]                               # [m,2]

    @torch.no_grad()
    def fno_pix(params_row):
        inp_np, scal_np = params_to_fieldinput(params_row[None], coords, side)
        f = fno((torch.tensor(inp_np, device=DEV) - im) / istd,
                (torch.tensor(scal_np, device=DEV) - sm) / sstd) * ostd + om
        d = f[0].permute(1, 2, 0).reshape(-1, 3)
        mk = sample_field_to_markers(d.view(1, side, side, 3).permute(0, 3, 1, 2), dense_t, sensor_t)
        return cam.project(deformed_marker_xyz(sensor_t, mk))[0]                               # [m,2]

    # sample test frames with real shear (mode>=1), spread across the set
    smag = np.hypot(params[:, 4], params[:, 5])
    cand = np.where((mode >= 1) & (smag > 1e-4))[0]
    if len(cand) > args.n_frames:
        cand = cand[np.linspace(0, len(cand) - 1, args.n_frames).round().astype(int)]
    print(f"evaluating {len(cand)} sheared frames, {args.noise_seeds} noise realisations each")

    # precompute true GT marker pixels + FNO marker pixels per frame
    pix_gt = torch.stack([field_to_pix(disp[i]) for i in cand])      # [F,m,2]
    pix_fn = torch.stack([fno_pix(params[i]) for i in cand])         # [F,m,2]
    F_ = pix_gt.shape[0]

    # ---- FNO model error in sensor space (clean, no noise) ----
    model_epe = float(torch.linalg.norm(pix_fn - pix_gt, dim=-1).mean())

    # ---- noise-floor sweep: localise the TRUE GT markers under camera noise ----
    # Decompose localisation error PER MARKER across noise realisations into:
    #   jitter (random std)        -> the NOISE floor (limits flow resolution; rises with noise)
    #   tracker bias (systematic)  -> centroid sub-pixel resolution, present even noise-free
    # Averaging EPE over markers first would wash out the random jitter (CLT), so we keep it
    # per-marker and aggregate the std.
    levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    S = args.noise_seeds
    floor_curve = {}
    for lv in levels:
        Tk = []
        for s in range(S):
            torch.manual_seed(1000 * s + int(lv * 1e4) + 1)
            imgs = render_dots(pix_gt, args.px, args.px, args.sigma, **render_kw)   # [F,1,H,W]
            noisy = add_camera_noise(imgs, photons=args.photons, read_noise=lv, blur=args.blur)
            Tk.append(track_flow_image(noisy, pix_gt, win=args.track_win, dark=True))  # [F,m,2] seed at truth
        Tk = torch.stack(Tk)                                  # [S,F,m,2]
        jit = torch.sqrt((Tk.std(0) ** 2).sum(-1))            # [F,m] random jitter (px)
        bia = torch.linalg.norm(Tk.mean(0) - pix_gt, dim=-1)  # [F,m] systematic tracker bias (px)
        floor_curve[lv] = {"jitter": float(jit.mean()), "jitter_p95": float(jit.flatten().quantile(0.95)),
                           "bias": float(bia.mean())}
        print(f"  read_noise={lv:.3f}  noise jitter={jit.mean():.3f}px (p95 {jit.flatten().quantile(0.95):.3f})  "
              f"tracker bias={bia.mean():.3f}px")

    op_jitter = floor_curve[args.read_noise_op]["jitter"]
    tracker_res = floor_curve[0.0]["bias"]      # noise-free systematic centroid resolution

    # ---- combined: what a controller actually sees = FNO(clean flow) vs noisy-tracked GT flow ----
    real_errs = []
    for s in range(args.noise_seeds):
        torch.manual_seed(7000 + s)
        rest_noisy = add_camera_noise(render_dots(pix_rest.expand(1, m, 2), args.px, args.px, args.sigma, **render_kw),
                                      photons=args.photons, read_noise=args.read_noise_op, blur=args.blur)
        tr_rest = track_flow_image(rest_noisy, pix_rest, win=args.track_win, dark=True)[0]   # [m,2]
        imgs = render_dots(pix_gt, args.px, args.px, args.sigma, **render_kw)
        noisy = add_camera_noise(imgs, photons=args.photons, read_noise=args.read_noise_op, blur=args.blur)
        tr_def = track_flow_image(noisy, pix_gt, win=args.track_win, dark=True)              # [F,m,2]
        flow_sensor = tr_def - tr_rest[None]                       # what the real readout gives
        flow_fno = pix_fn - pix_rest                               # FNO clean prediction
        real_errs.append(torch.linalg.norm(flow_fno - flow_sensor, dim=-1).mean().item())
    real_epe = float(np.mean(real_errs))

    # crossover: smallest sweep level whose random jitter alone reaches the FNO model error
    crossover = next((lv for lv in levels if floor_curve[lv]["jitter"] >= model_epe), None)
    if op_jitter >= model_epe:
        verdict = (f"camera noise jitter {op_jitter:.2f}px (@ {args.read_noise_op:.0%}) reaches the FNO error "
                   f"{model_epe:.2f}px -> sensor noise masks the surrogate error")
    else:
        verdict = (f"camera noise jitter {op_jitter:.2f}px (@ {args.read_noise_op:.0%}) is FAR below the FNO error "
                   f"{model_epe:.2f}px -> sensor noise does NOT mask it; the limiting factors are FNO accuracy "
                   f"({model_epe:.2f}px) and the centroid-tracker resolution ({tracker_res:.2f}px), both >> noise")
    print(f"\nmodel EPE (clean)      = {model_epe:.3f} px   ({100*model_epe/pitch_px:.0f}% of pitch {pitch_px:.1f}px)")
    print(f"noise jitter @ {args.read_noise_op:.0%}    = {op_jitter:.3f} px   (random, per-marker)")
    print(f"tracker resolution     = {tracker_res:.3f} px   (systematic, noise-free centroid)")
    print(f"combined readout EPE   = {real_epe:.3f} px   (FNO clean vs noisy-tracked GT flow)")
    print(f"jitter reaches model at read_noise = {crossover}")
    print(f"VERDICT: {verdict}")

    rep = {"data": args.data, "n_frames": int(F_), "noise_seeds": args.noise_seeds,
           "pitch_px": pitch_px, "model_epe_px": model_epe,
           "operating_point": {"photons": args.photons, "blur_px": args.blur, "read_noise": args.read_noise_op},
           "noise_floor_curve": {f"{k:.3f}": v for k, v in floor_curve.items()},
           "noise_jitter_op_px": op_jitter, "tracker_resolution_px": tracker_res,
           "combined_readout_epe_px": real_epe,
           "noise_masks_model": bool(op_jitter >= model_epe), "jitter_crossover_read_noise": crossover,
           "verdict": verdict}

    out_dir = RUNS / "phase6"; ensure(out_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (1) noise jitter (random) vs read noise, against FNO error + tracker resolution
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        lv = np.array(levels) * 100
        jit = np.array([floor_curve[l]["jitter"] for l in levels])
        j95 = np.array([floor_curve[l]["jitter_p95"] for l in levels])
        ax.plot(lv, jit, "-o", ms=5, label="noise jitter (random, mean)")
        ax.plot(lv, j95, ":^", ms=4, color="C0", alpha=0.6, label="noise jitter (p95 marker)")
        ax.axhline(model_epe, color="crimson", ls="--", label=f"FNO model error {model_epe:.2f}px")
        ax.axhline(tracker_res, color="green", ls="-.", label=f"centroid-tracker resolution {tracker_res:.2f}px")
        ax.axvline(args.read_noise_op * 100, color="gray", ls=":", alpha=0.7,
                   label=f"operating point {args.read_noise_op:.0%}")
        ax.set_xlabel("camera read noise (% full-scale)"); ax.set_ylabel("marker EPE (px)")
        ax.set_title("Sensor noise jitter vs FNO surrogate error")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "realism_floor.png", dpi=130); plt.close(fig)
        rep["floor_plot"] = str(out_dir / "realism_floor.png")
        print(f"saved {out_dir/'realism_floor.png'}")

        # (2) clean vs noisy dot image montage on a high-shear frame
        hi = int(cand[np.argmax(smag[cand])])
        pg = field_to_pix(disp[hi])[None]
        torch.manual_seed(0)
        clean = render_dots(pg, args.px, args.px, args.sigma, **render_kw)
        noisy = add_camera_noise(clean, photons=args.photons, read_noise=args.read_noise_op, blur=args.blur)
        fig2, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
        for ax, img, ttl in [(axes[0], clean, "noiseless render"),
                             (axes[1], noisy, f"+camera noise ({args.read_noise_op:.0%}, blur {args.blur}px)")]:
            ax.imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig2.suptitle(f"Marker-dot image: clean vs realistic camera "
                      f"(noise jitter {op_jitter:.2f}px, tracker res {tracker_res:.2f}px)", fontsize=11)
        fig2.tight_layout(); fig2.savefig(out_dir / "realism_image.png", dpi=130); plt.close(fig2)
        rep["image_plot"] = str(out_dir / "realism_image.png")
        print(f"saved {out_dir/'realism_image.png'}")
    except Exception as e:
        rep["plot_error"] = str(e)
        print(f"plot skipped: {e}")

    json.dump(rep, open(out_dir / "realism.json", "w"), indent=2, default=float)
    print(f"saved {out_dir/'realism.json'}")


if __name__ == "__main__":
    main()
