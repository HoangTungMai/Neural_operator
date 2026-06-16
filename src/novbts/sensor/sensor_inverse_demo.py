#!/usr/bin/env python3
"""Phase 5b -- the integrated Track-A <-> Track-B payoff.

(1) Compatibility: the FNO surrogate + the differentiable renderer reproduce the
    sensor observation -- FEM disp -> render vs FNO(contact) -> render agree in
    marker-flow space over the held-out set.
(2) Differentiable inverse FROM THE SENSOR IMAGE: recover the applied shear (sx,sy)
    of a frame by gradient descent through  render . FNO  on the rendered marker
    image (not the raw displacement field). Gradients flow image <- renderer <- FNO
    <- action, so the whole sensor pipeline is differentiable end-to-end.

  python -m novbts.sensor.sensor_inverse_demo --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json
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
        else:
            locs = np.linspace(0.22, 0.86, n_per_mode)
            locs = np.clip(np.round(locs * (len(order) - 1)).astype(int), 0, len(order) - 1)
            frames = order[locs]
        picked.extend((int(f), name) for f in frames)
    return picked


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
    ap.add_argument("--compare-n-per-mode", type=int, default=2,
                    help="GT-vs-FNO visual comparison samples per contact mode")
    ap.add_argument("--skip-inverse", action="store_true",
                    help="only train FNO and render GT-vs-FNO comparison, skip image inverse optimization")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--opt-lr", type=float, default=0.05)
    args = ap.parse_args()

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
        compat = {"flow_rel_l2_overall": grel(msel >= 0),
                  "flow_rel_l2_stick": grel(msel <= 1),
                  "flow_rel_l2_slip": grel(msel >= 2)}
    print("=== (1) FNO+renderer reproduces the sensor observation (marker-flow rel L2) ===")
    print(f"  overall={compat['flow_rel_l2_overall']:.3f}  stick={compat['flow_rel_l2_stick']:.3f}  "
          f"slip={compat['flow_rel_l2_slip']:.3f}")

    phase_dir = RUNS / "phase5"; ensure(phase_dir)
    flow_score = torch.linalg.norm(flow_fem, dim=-1).amax(dim=1).cpu().numpy()
    local_samples = representative_frames(msel.cpu().numpy(), flow_score, args.compare_n_per_mode)
    compare = {"frames": [], "image_mse": [], "flow_rel_l2": [], "compat": compat,
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
            fig, axes = plt.subplots(nrows, 4, figsize=(12.5, 2.8 * nrows), squeeze=False)
            im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
            diff_max = max(float(img_err.max()), 1e-6)
            pr_np = pix_rest[0].cpu().numpy()
            for r in range(nrows):
                frame_i = compare["frames"][r]["frame"]
                mode_name = compare["frames"][r]["mode"]
                gt_np = img_gt[r, 0].cpu().numpy()
                pred_np = img_pred[r, 0].cpu().numpy()
                err_np = img_err[r, 0].cpu().numpy()
                res_np = flow_err[r].cpu().numpy()
                axes[r, 0].imshow(gt_np, **im_kw)
                axes[r, 0].set_title(f"GT image\n{mode_name} frame {frame_i}", fontsize=9)
                axes[r, 1].imshow(pred_np, **im_kw)
                axes[r, 1].set_title("FNO image", fontsize=9)
                axes[r, 2].imshow(err_np, cmap="magma", vmin=0.0, vmax=diff_max, interpolation="none")
                axes[r, 2].set_title(f"|GT-FNO|\nMSE={compare['image_mse'][r]:.2e}", fontsize=9)
                axes[r, 3].imshow(gt_np, **im_kw)
                axes[r, 3].quiver(pr_np[:, 0], pr_np[:, 1], res_np[:, 0], -res_np[:, 1],
                                  color="cyan", scale_units="xy", angles="xy", scale=0.35, width=0.004)
                axes[r, 3].set_title(f"flow residual\nrelL2={compare['flow_rel_l2'][r]:.2f}", fontsize=9)
                for ax in axes[r]:
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlim(0, args.px); ax.set_ylim(args.px, 0)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
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
    sx_t, sy_t = float(D["params"][fi, 4]), float(D["params"][fi, 5])
    pen, mask, _ = build_input_channels(D["params"][fi], coords, side)
    pen3, mask1 = pen[None, None], mask[None, None]
    scal_i = nsc(scal[fi:fi + 1])
    with torch.no_grad():
        img_obs = render_dots(disp_to_pix(out[fi:fi + 1]), args.px, args.px, args.sigma,
                              **render_kw)   # the observed sensor image

    def render_from_action(sx, sy):
        ch1 = sx * mask1; ch2 = sy * mask1
        inp3 = torch.cat([pen3, ch1, ch2], 1)
        field = fno((inp3 - im) / istd, scal_i) * ostd + om
        return render_dots(disp_to_pix(field), args.px, args.px, args.sigma, **render_kw)

    s_abs = float(np.abs(D["params"][:, 4:6]).max())
    v = torch.zeros(2, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([v], lr=args.opt_lr * s_abs)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = ((render_from_action(v[0], v[1]) - img_obs) ** 2).mean()
        loss.backward(); opt.step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sx_r, sy_r = float(v.detach()[0]), float(v.detach()[1])
    true = np.array([sx_t, sy_t]); got = np.array([sx_r, sy_r])
    rel_err = float(np.linalg.norm(got - true) / (np.linalg.norm(true) + 1e-9))
    mag_t = float(np.hypot(sx_t, sy_t)); ang_t = float(np.degrees(np.arctan2(sy_t, sx_t)))
    ang_r = float(np.degrees(np.arctan2(sy_r, sx_r)))

    print(f"\n=== (2) recover shear from the rendered marker IMAGE (autograd thru render.FNO) ===")
    print(f"observed frame {fi} ({MODE_NAMES[int(mode[fi])]})  |s|={mag_t*1e3:.3f}mm @ {ang_t:.1f}deg")
    print(f"true       (sx,sy)=({sx_t*1e3:8.3f},{sy_t*1e3:8.3f}) mm")
    print(f"recovered  (sx,sy)=({sx_r*1e3:8.3f},{sy_r*1e3:8.3f}) mm   rel_err={rel_err:.3f}  "
          f"dir_err={abs(ang_r-ang_t):.1f}deg  ({wall:.1f}s)")
    print(f"final image loss={loss.item():.3e}  (gradients flowed image<-renderer<-FNO<-action)")

    rep = {"data": args.data, "px": args.px, "sensor_M": int(sensor_M),
           "sensor_marker_side": args.sensor_marker_side, "marker_placement": args.marker_placement,
           "marker_pixel_fill": args.marker_pixel_fill, "marker_inset": args.marker_inset,
           "camera": cam.as_dict(),
           "dot_style": {"polarity": args.dot_polarity, "background": args.background,
                         "contrast": args.contrast, "saturate": args.saturate_dots},
           "compat": compat,
           "inverse": {"frame": fi, "mode": MODE_NAMES[int(mode[fi])],
                       "true_shear_mm": [sx_t * 1e3, sy_t * 1e3],
                       "recovered_shear_mm": [sx_r * 1e3, sy_r * 1e3],
                       "rel_err": rel_err, "dir_err_deg": abs(ang_r - ang_t),
                       "final_image_loss": float(loss.item()), "wall_s": round(wall, 2),
                       "steps": args.steps}}
    json.dump(rep["compat"], open(phase_dir / "sensor_compat.json", "w"), indent=2, default=float)
    json.dump(rep["inverse"], open(phase_dir / "sensor_inverse.json", "w"), indent=2, default=float)
    json.dump(compare, open(phase_dir / "sensor_compare.json", "w"), indent=2, default=float)
    print(f"\nSaved {phase_dir/'sensor_compat.json'} + sensor_compare.json + sensor_inverse.json")


if __name__ == "__main__":
    main()
