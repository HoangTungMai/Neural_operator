#!/usr/bin/env python3
"""Phase 5b -- the integrated Track-A <-> Track-B payoff.

(1) Compatibility: the FNO surrogate + the differentiable renderer reproduce the
    sensor observation -- FEM disp -> render vs FNO(contact) -> render agree in
    marker-flow space over the held-out set.
(2) Differentiable inverse FROM THE SENSOR IMAGE: recover the applied shear (sx,sy)
    of a frame by gradient descent through  render . FNO  on the rendered marker
    image (not the raw displacement field). Gradients flow image <- renderer <- FNO
    <- action, so the whole sensor pipeline is differentiable end-to-end.

  python -m novbts.sensor.sensor_inverse_demo \
    --data data/uipc/shear_res24_avg_swept_REALISTIC.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from novbts.operator.field2field import FNOField, train_operator, count_parameters, DEV
from novbts.operator.fem_benchmark import load, norm_from
from novbts.operator.inverse_demo import build_input_channels
from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, marker_half_extent,
    sensor_marker_grid, sensor_marker_grid_pixel_even, sample_field_to_markers,
)
from novbts.paths import FEM, RUNS, ensure


def representative_frames(mode, score, n_per_mode):
    picked = []
    for m, name in enumerate(MODE_NAMES):
        idx = np.where(mode == m)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(score[idx])]
        if len(order) <= n_per_mode:
            frames = order
        elif n_per_mode == 1:
            # One visual sample per mode should be representative but visually
            # readable; avoid the low-shear tail where image differences vanish.
            frames = order[[int(round(0.70 * (len(order) - 1)))]]
        else:
            locs = np.linspace(0.22, 0.86, n_per_mode)
            locs = np.clip(np.round(locs * (len(order) - 1)).astype(int), 0, len(order) - 1)
            # Rounded quantiles can collide when a mode has only slightly more
            # candidates than requested. Keep frames unique and fill from the
            # high-score end, where shear/image direction is better defined.
            unique_locs = list(dict.fromkeys(locs.tolist()))
            for loc in range(len(order) - 1, -1, -1):
                if len(unique_locs) >= n_per_mode:
                    break
                if loc not in unique_locs:
                    unique_locs.append(loc)
            frames = order[np.sort(unique_locs[:n_per_mode])]
        picked.extend((int(f), name) for f in frames)
    return picked


def circular_direction_error_deg(recovered_deg, true_deg):
    """Smallest absolute angular difference, in [0, 180] degrees."""
    return float(abs((recovered_deg - true_deg + 180.0) % 360.0 - 180.0))


def inverse_summary(rows):
    """Aggregate per-frame image-inversion metrics with population std."""
    mag_pct = np.asarray([100.0 * row["mag_rel_err"] for row in rows], dtype=np.float64)
    direction = np.asarray([row["dir_err_deg"] for row in rows], dtype=np.float64)

    def stats(values):
        if len(values) == 0:
            return {"mean": None, "std": None}
        return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    return {
        "n": int(len(rows)),
        "magnitude_error_pct": stats(mag_pct),
        "direction_error_deg": stats(direction),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--dot-polarity", choices=["bright", "dark"], default="dark")
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    ap.add_argument("--saturate-dots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compare-n-per-mode", type=int, default=1,
                    help="GT-vs-FNO visual comparison samples per contact mode")
    ap.add_argument("--skip-inverse", action="store_true",
                    help="only train FNO and render GT-vs-FNO comparison, skip image inverse optimization")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--inverse-n-per-mode", type=int, default=5,
                    help="representative image-inverse frames per contact mode")
    ap.add_argument("--inverse-restarts", type=int, default=8,
                    help="fixed initialisations per frame; 1 reproduces zero-init behavior")
    ap.add_argument("--inverse-min-shear-frac", type=float, default=0.01,
                    help="exclude direction-ill-defined frames below this fraction "
                         "of the maximum test-set shear")
    args = ap.parse_args()
    if args.inverse_n_per_mode < 1:
        ap.error("--inverse-n-per-mode must be at least 1")
    if args.inverse_restarts < 1:
        ap.error("--inverse-restarts must be at least 1")
    if not 0.0 <= args.inverse_min_shear_frac < 1.0:
        ap.error("--inverse-min-shear-frac must be in [0, 1)")
    if args.steps < 1:
        ap.error("--steps must be at least 1")

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    coords = np.load(args.data, allow_pickle=True)["coords"]
    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    tr = torch.arange(0, N - nt, device=DEV); te = torch.arange(N - nt, N, device=DEV)
    print(f"device={DEV}  sensor inverse  data={args.data}  N={N} side={side} px={args.px}")

    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    secs, _ = train_operator(fno, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]), mode[tr], cg, args.epochs, args.lr)
    fno_params = count_parameters(fno)
    fno.eval()
    for p in fno.parameters():
        p.requires_grad_(False)
    print(f"[FNO] trained {secs:.0f}s  ({fno_params} params)  [frozen]\n")

    # --- sensor pieces ---
    cam = PinholeCamera.from_gel(marker_half_extent(coords), px=args.px, working_dist=args.working_dist)
    if args.marker_placement == "pixel_even":
        sensor_coords = sensor_marker_grid_pixel_even(
            cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
    else:
        sensor_coords = sensor_marker_grid(coords, args.sensor_marker_side, inset=args.marker_inset)
    field_coords_t = torch.tensor(coords, device=DEV)
    coords_t = torch.tensor(sensor_coords, device=DEV)
    sensor_M = sensor_coords.shape[0]
    pix_rest = cam.project(deformed_marker_xyz(coords_t, torch.zeros(1, sensor_M, 3, device=DEV)))   # [1,m,2]
    render_kw = dict(background=args.background, contrast=args.contrast,
                     polarity=args.dot_polarity, saturate=args.saturate_dots)

    def disp_to_pix(field):                       # field [B,3,H,W] -> pix [B,M,2]
        markers = sample_field_to_markers(field, field_coords_t, coords_t)
        return cam.project(deformed_marker_xyz(coords_t, markers))

    # ===== (1) compatibility: FEM-render vs FNO-render in marker-flow space =====
    with torch.no_grad():
        fno_te = fno(nin(inp[te]), nsc(scal[te])) * ostd + om   # [nt,3,H,W] RAW disp (denormalised)
        flow_fem = disp_to_pix(out[te]) - pix_rest
        flow_fno = disp_to_pix(fno_te) - pix_rest
        msel = mode[te]
        # aggregate rel-L2 per regime (norm of all residuals / norm of all flow) -- avoids the
        # per-frame blow-up on near-zero-flow normal frames (in-plane flow ~0 there).
        def grel(sel):
            a = (flow_fno[sel] - flow_fem[sel]).reshape(-1)
            b = flow_fem[sel].reshape(-1)
            return float(a.norm() / (b.norm() + 1e-9))
        compat = {"gt": os.path.basename(args.data), "gt_path": args.data,
                  "flow_rel_l2_overall": grel(msel >= 0),
                  "flow_rel_l2_stick": grel(msel <= 1),
                  "flow_rel_l2_slip": grel(msel >= 2)}
    print("=== (1) FNO+renderer reproduces the sensor observation (marker-flow rel L2) ===")
    print(f"  overall={compat['flow_rel_l2_overall']:.3f}  stick={compat['flow_rel_l2_stick']:.3f}  "
          f"slip={compat['flow_rel_l2_slip']:.3f}")

    phase_dir = RUNS / "phase5"; ensure(phase_dir)
    flow_score = torch.linalg.norm(flow_fem, dim=-1).amax(dim=1).cpu().numpy()
    local_samples = representative_frames(msel.cpu().numpy(), flow_score, args.compare_n_per_mode)
    compare = {"gt": os.path.basename(args.data), "gt_path": args.data,
               "frames": [], "image_mse": [], "flow_rel_l2": [], "compat": compat,
               "epochs": args.epochs}
    if local_samples:
        local_idx = torch.tensor([i for i, _ in local_samples], device=DEV)
        global_idx = te[local_idx]
        labels = [name for _, name in local_samples]
        with torch.no_grad():
            pix_gt = disp_to_pix(out[global_idx])
            pix_pred = disp_to_pix(fno_te[local_idx])
            img_gt = render_dots(pix_gt, args.px, args.px, args.sigma, **render_kw)
            img_pred = render_dots(pix_pred, args.px, args.px, args.sigma, **render_kw)
            flow_gt = pix_gt - pix_rest
            flow_pred = pix_pred - pix_rest
        img_err = (img_pred - img_gt).abs()
        flow_err = flow_pred - flow_gt
        for row_i, frame_i in enumerate(global_idx.cpu().numpy().tolist()):
            compare["frames"].append({"frame": int(frame_i), "mode": labels[row_i]})
            compare["image_mse"].append(float(((img_pred[row_i] - img_gt[row_i]) ** 2).mean()))
            num = torch.linalg.norm(flow_err[row_i].reshape(-1))
            den = torch.linalg.norm(flow_gt[row_i].reshape(-1)) + 1e-9
            compare["flow_rel_l2"].append(float(num / den))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            nrows = len(local_samples)
            fig, axes = plt.subplots(nrows, 4, figsize=(7.2, 1.85 * nrows), squeeze=False)
            im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
            diff_max = max(float(img_err.max()), 1e-6)
            pr_np = pix_rest[0].cpu().numpy()
            col_titles = ("GT image", "FNO image", "|GT-FNO| image", "flow residual")
            for r in range(nrows):
                frame_i = compare["frames"][r]["frame"]
                mode_name = compare["frames"][r]["mode"]
                gt_np = img_gt[r, 0].cpu().numpy()
                pred_np = img_pred[r, 0].cpu().numpy()
                err_np = img_err[r, 0].cpu().numpy()
                res_np = flow_err[r].cpu().numpy()
                axes[r, 0].imshow(gt_np, **im_kw)
                axes[r, 0].set_ylabel(f"{mode_name}\n#{frame_i}", fontsize=8)
                axes[r, 1].imshow(pred_np, **im_kw)
                axes[r, 2].imshow(err_np, cmap="magma", vmin=0.0, vmax=diff_max, interpolation="none")
                axes[r, 2].text(0.04, 0.93, f"MSE={compare['image_mse'][r]:.1e}",
                                transform=axes[r, 2].transAxes, color="white",
                                fontsize=7, va="top")
                axes[r, 3].imshow(gt_np, **im_kw)
                axes[r, 3].quiver(pr_np[:, 0], pr_np[:, 1], res_np[:, 0], -res_np[:, 1],
                                  color="cyan", scale_units="xy", angles="xy", scale=0.35, width=0.005)
                axes[r, 3].text(0.04, 0.93, f"relL2={compare['flow_rel_l2'][r]:.2f}",
                                transform=axes[r, 3].transAxes, color="black",
                                fontsize=7, va="top",
                                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
                for ax in axes[r]:
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0)
                    ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    for c, title in enumerate(col_titles):
                        axes[r, c].set_title(title, fontsize=8.5)
            fig.tight_layout(pad=0.35, h_pad=0.45, w_pad=0.35)
            fig.savefig(phase_dir / "gt_vs_fno_samples.png", dpi=150)
            plt.close(fig)
            compare["plot"] = str(phase_dir / "gt_vs_fno_samples.png")
            print(f"saved GT-vs-FNO samples -> {phase_dir / 'gt_vs_fno_samples.png'}")
        except Exception as e:
            compare["plot_error"] = str(e)
            print(f"GT-vs-FNO plot skipped: {e}")

    json.dump(compare, open(phase_dir / "sensor_compare.json", "w"), indent=2, default=float)
    if args.skip_inverse:
        print(f"\nSaved {phase_dir/'sensor_compare.json'}")
        return

    json.dump(compat, open(phase_dir / "sensor_compat.json", "w"), indent=2, default=float)
    # ===== (2) inverse from the rendered sensor IMAGE via autograd through render . FNO =====
    cand = te[mode[te] == 3]
    if len(cand) == 0:
        cand = te[mode[te] >= 2]
    smag = np.hypot(D["params"][cand.cpu().numpy(), 4], D["params"][cand.cpu().numpy(), 5])
    fi = int(cand[int(np.argmax(smag))])
    s_abs = float(np.abs(D["params"][:, 4:6]).max())
    te_np = te.cpu().numpy()
    te_mode = mode[te].cpu().numpy()
    te_smag = np.linalg.norm(D["params"][te_np, 4:6], axis=1)
    positive_test_shear = te_smag[te_smag > 0.0]
    restart_radius = float(np.median(positive_test_shear)) if len(positive_test_shear) else s_abs
    restart_inits = np.zeros((args.inverse_restarts, 2), dtype=np.float32)
    if args.inverse_restarts > 1:
        restart_angles = (
            2.0 * np.pi * np.arange(args.inverse_restarts - 1)
            / (args.inverse_restarts - 1)
        )
        restart_inits[1:, 0] = restart_radius * np.cos(restart_angles)
        restart_inits[1:, 1] = restart_radius * np.sin(restart_angles)

    def invert_frame(frame_i):
        sx_t, sy_t = float(D["params"][frame_i, 4]), float(D["params"][frame_i, 5])
        pen, mask, _ = build_input_channels(D["params"][frame_i], coords, side)
        pen3 = pen[None, None].expand(args.inverse_restarts, -1, -1, -1)
        mask1 = mask[None, None]
        scal_i = nsc(scal[frame_i:frame_i + 1]).expand(args.inverse_restarts, -1)
        with torch.no_grad():
            img_obs = render_dots(
                disp_to_pix(out[frame_i:frame_i + 1]),
                args.px, args.px, args.sigma, **render_kw)

        def render_from_action(actions):
            ch1 = actions[:, 0, None, None, None] * mask1
            ch2 = actions[:, 1, None, None, None] * mask1
            inp3 = torch.cat([pen3, ch1, ch2], 1)
            field = fno((inp3 - im) / istd, scal_i) * ostd + om
            return render_dots(disp_to_pix(field), args.px, args.px, args.sigma, **render_kw)

        v = torch.tensor(restart_inits, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([v], lr=args.opt_lr * s_abs)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            restart_loss = ((render_from_action(v) - img_obs) ** 2).flatten(1).mean(1)
            restart_loss.sum().backward()
            opt.step()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        all_recovered = v.detach().cpu().numpy().astype(np.float64)
        restart_losses = restart_loss.detach().cpu().numpy().astype(np.float64)
        winning_restart_idx = int(np.argmin(restart_losses))
        recovered = all_recovered[winning_restart_idx]
        true = np.asarray([sx_t, sy_t], dtype=np.float64)
        mag_t = float(np.linalg.norm(true))
        mag_r = float(np.linalg.norm(recovered))
        ang_t = float(np.degrees(np.arctan2(sy_t, sx_t)))
        ang_r = float(np.degrees(np.arctan2(recovered[1], recovered[0])))
        return {
            "frame": int(frame_i),
            "mode": MODE_NAMES[int(mode[frame_i])],
            "true_shear_mm": (true * 1e3).tolist(),
            "recovered_shear_mm": (recovered * 1e3).tolist(),
            "true_magnitude_mm": mag_t * 1e3,
            "recovered_magnitude_mm": mag_r * 1e3,
            # Preserve the legacy vector-relative metric and add the requested
            # magnitude-only relative error for the multi-frame aggregate.
            "rel_err": float(np.linalg.norm(recovered - true) / (mag_t + 1e-9)),
            "mag_rel_err": float(abs(mag_r - mag_t) / (mag_t + 1e-9)),
            "dir_err_deg": circular_direction_error_deg(ang_r, ang_t),
            "final_image_loss": float(restart_losses[winning_restart_idx]),
            "wall_s": round(wall, 2),
            "steps": args.steps,
            "n_restarts": args.inverse_restarts,
            "winning_restart_idx": winning_restart_idx,
            "restart_losses": restart_losses.tolist(),
            "restart_initial_shear_mm": (restart_inits.astype(np.float64) * 1e3).tolist(),
            "restart_recovered_shear_mm": (all_recovered * 1e3).tolist(),
        }

    legacy = invert_frame(fi)
    legacy.update(gt=os.path.basename(args.data), gt_path=args.data)
    sx_t, sy_t = legacy["true_shear_mm"]
    sx_r, sy_r = legacy["recovered_shear_mm"]
    mag_t = legacy["true_magnitude_mm"]
    ang_t = float(np.degrees(np.arctan2(sy_t, sx_t)))

    print(f"\n=== (2) recover shear from the rendered marker IMAGE (autograd thru render.FNO) ===")
    print(f"observed frame {fi} ({legacy['mode']})  |s|={mag_t:.3f}mm @ {ang_t:.1f}deg")
    print(f"true       (sx,sy)=({sx_t:8.3f},{sy_t:8.3f}) mm")
    print(f"recovered  (sx,sy)=({sx_r:8.3f},{sy_r:8.3f}) mm   "
          f"rel_err={legacy['rel_err']:.3f}  dir_err={legacy['dir_err_deg']:.1f}deg  "
          f"restart={legacy['winning_restart_idx']}/{legacy['n_restarts'] - 1}  "
          f"({legacy['wall_s']:.1f}s)")
    print(f"final image loss={legacy['final_image_loss']:.3e}  "
          f"(gradients flowed image<-renderer<-FNO<-action)")

    # ===== (3) aggregate image inversion over representative frames per mode =====
    min_shear = float(args.inverse_min_shear_frac * te_smag.max())
    eligible_mode = te_mode.copy()
    eligible_mode[te_smag < min_shear] = -1
    local_inverse = representative_frames(
        eligible_mode, te_smag, args.inverse_n_per_mode)

    inverse_started = time.perf_counter()
    inverse_rows = []
    for row_i, (local_i, mode_name) in enumerate(local_inverse, start=1):
        frame_i = int(te_np[local_i])
        row = invert_frame(frame_i)
        inverse_rows.append(row)
        print(f"[inverse {row_i:02d}/{len(local_inverse):02d}] frame={frame_i} "
              f"mode={mode_name:12s} mag={100.0 * row['mag_rel_err']:6.2f}% "
              f"dir={row['dir_err_deg']:6.2f}deg "
              f"win={row['winning_restart_idx']}/{row['n_restarts'] - 1} "
              f"wall={row['wall_s']:.1f}s")
    inverse_wall = time.perf_counter() - inverse_started

    overall = inverse_summary(inverse_rows)
    by_mode = {
        name: inverse_summary([row for row in inverse_rows if row["mode"] == name])
        for name in MODE_NAMES
    }
    multiframe = {
        "gt": os.path.basename(args.data),
        "gt_path": args.data,
        "data": args.data,
        "dataset_frames": int(N),
        "train_frames": int(N - nt),
        "test_frames": int(nt),
        "device": str(DEV),
        "epochs": args.epochs,
        "steps": args.steps,
        "opt_lr": args.opt_lr,
        "inverse_n_per_mode_requested": args.inverse_n_per_mode,
        "inverse_restarts": args.inverse_restarts,
        "restart_init_radius_mm": restart_radius * 1e3,
        "restart_init_scheme": "zero plus evenly spaced fixed directions",
        "inverse_min_shear_frac": args.inverse_min_shear_frac,
        "inverse_min_shear_mm": min_shear * 1e3,
        "selection": "representative quantiles 0.22--0.86 by test-frame shear magnitude",
        "frames": inverse_rows,
        "overall": overall,
        "by_mode": by_mode,
        "inverse_wall_s": round(inverse_wall, 2),
        "fno_train_wall_s": round(float(secs), 2),
    }

    print("\n=== (3) multi-frame image inversion: mean +/- population std ===")
    for label, summary in [("overall", overall), *by_mode.items()]:
        mag = summary["magnitude_error_pct"]
        direction = summary["direction_error_deg"]
        if summary["n"]:
            print(f"{label:12s} n={summary['n']:2d}  "
                  f"magnitude={mag['mean']:.2f}+/-{mag['std']:.2f}%  "
                  f"direction={direction['mean']:.2f}+/-{direction['std']:.2f}deg")
        else:
            print(f"{label:12s} n= 0  (no frame above shear threshold)")

    rep = {"gt": os.path.basename(args.data), "gt_path": args.data,
           "data": args.data, "px": args.px, "sensor_M": int(sensor_M),
           "sensor_marker_side": args.sensor_marker_side, "marker_placement": args.marker_placement,
           "marker_pixel_fill": args.marker_pixel_fill, "marker_inset": args.marker_inset,
           "camera": cam.as_dict(),
           "dot_style": {"polarity": args.dot_polarity, "background": args.background,
                         "contrast": args.contrast, "saturate": args.saturate_dots},
           "compat": compat,
           "inverse": {"gt": os.path.basename(args.data), "gt_path": args.data,
                       "frame": legacy["frame"], "mode": legacy["mode"],
                       "true_shear_mm": legacy["true_shear_mm"],
                       "recovered_shear_mm": legacy["recovered_shear_mm"],
                       "rel_err": legacy["rel_err"], "dir_err_deg": legacy["dir_err_deg"],
                       "final_image_loss": legacy["final_image_loss"],
                       "wall_s": legacy["wall_s"], "steps": args.steps,
                       "n_restarts": legacy["n_restarts"],
                       "winning_restart_idx": legacy["winning_restart_idx"],
                       "restart_losses": legacy["restart_losses"],
                       "restart_initial_shear_mm": legacy["restart_initial_shear_mm"],
                       "restart_recovered_shear_mm": legacy["restart_recovered_shear_mm"]}}
    json.dump(rep["compat"], open(phase_dir / "sensor_compat.json", "w"), indent=2, default=float)
    json.dump(rep["inverse"], open(phase_dir / "sensor_inverse.json", "w"), indent=2, default=float)
    json.dump(multiframe, open(phase_dir / "sensor_inverse_multiframe.json", "w"),
              indent=2, default=float)
    json.dump(compare, open(phase_dir / "sensor_compare.json", "w"), indent=2, default=float)
    print(f"\nSaved {phase_dir/'sensor_compat.json'} + sensor_compare.json + "
          "sensor_inverse.json + sensor_inverse_multiframe.json")


if __name__ == "__main__":
    main()
