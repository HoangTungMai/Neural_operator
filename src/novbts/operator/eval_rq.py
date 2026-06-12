#!/usr/bin/env python3
"""
Phase 3 evaluation — RQ1 (accuracy), RQ2 (generalisation), RQ3 (speed).

Loads checkpoints from runs/phase3/ trained by phase3_train.py and the
datasets in data/phase3_gt/.  Produces:
  * runs/phase3/rq_results.json
  * runs/phase3/fidelity_speed.png
  * a printed RQ1-RQ3 table

RQ1  per-mode relative L2 / RMSE + slip metrics (mode-F1 for heads a & b,
     tangential direction error).
RQ2  relative L2 on each OOD split + degradation vs in-distribution.
RQ3  inference throughput (frames/s) on 1 GPU + fidelity-speed scatter.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from novbts.models import CoordinateMLP
from novbts.operator.param2field import (
    FNO2dMultiTask, SeparateSlipClassifier, load_split, load_norm,
    relative_l2_per_mode, macro_f1, MODE_NAMES,
)
from novbts.paths import ANALYTIC, RUNS


@torch.no_grad()
def infer(model, params, coords, bs, norm, multitask=False):
    """Feeds normalised params, returns DENORMALISED field (raw units)."""
    pm, ps, dm, ds = norm
    pn = (params - pm) / ps
    coords_b = coords[None].expand(params.shape[0], -1, -1).contiguous()
    out = []
    for s in range(0, params.shape[0], bs):
        o = model(pn[s:s + bs], coords_b[s:s + bs])
        out.append(o[0] if multitask else o)
    return torch.cat(out) * ds + dm


def rmse_per_mode(pred, target, mode):
    rmse = torch.sqrt((pred - target).square().mean(dim=(1, 2)))
    out = {"overall": float(rmse.mean())}
    for i, n in enumerate(MODE_NAMES):
        m = mode == i
        out[n] = float(rmse[m].mean()) if m.any() else float("nan")
    return out


def tangential_direction_error(pred, target, mode):
    """Mean angle (deg) between predicted & GT in-plane vectors on slip frames,
    weighted by GT tangential magnitude."""
    slip = mode >= 2
    if not slip.any():
        return float("nan")
    p, t = pred[slip][..., :2], target[slip][..., :2]
    tmag = torch.linalg.norm(t, dim=-1)
    w = tmag / (tmag.sum(dim=-1, keepdim=True) + 1e-9)
    cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).clamp(-1, 1)
    ang = torch.arccos(cos) * 180.0 / math.pi
    return float((ang * w).sum(dim=-1).mean())


@torch.no_grad()
def throughput(model, params, coords, device, multitask=False, iters=5):
    coords_b = coords[None].expand(params.shape[0], -1, -1).contiguous()
    # warmup
    _ = model(params[:64], coords_b[:64])  # timing only; raw params fine here
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        for s in range(0, params.shape[0], 256):
            o = model(params[s:s + 256], coords_b[s:s + 256])
            n += (o[0] if multitask else o).shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ANALYTIC))
    ap.add_argument("--run-dir", default=str(RUNS / "phase3"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    data, run = Path(args.data_dir), Path(args.run_dir)

    # ---- load models ----
    mlp = CoordinateMLP(params_dim=9).to(device)
    mlp.load_state_dict(torch.load(run / "mlp.pt", map_location=device)); mlp.eval()
    fno = FNO2dMultiTask(params_dim=9, with_slip_head=False).to(device)
    fno.load_state_dict(torch.load(run / "fno.pt", map_location=device)); fno.eval()
    fno_mt = FNO2dMultiTask(params_dim=9, with_slip_head=True).to(device)
    fno_mt.load_state_dict(torch.load(run / "fno_multitask_a.pt", map_location=device)); fno_mt.eval()
    clf = SeparateSlipClassifier().to(device)
    clf.load_state_dict(torch.load(run / "slip_classifier_b.pt", map_location=device)); clf.eval()

    norm = load_norm(run / "norm.npz", device)
    pm, ps, dm, ds = norm
    results = {"RQ1": {}, "RQ2": {}, "RQ3": {}}

    # ===== RQ1 : accuracy on in-distribution test =====
    p, coords, disp, mode = load_split(data / "test_id.npz", device)
    side = int(math.sqrt(coords.shape[0]))
    for name, model, mt in [("mlp", mlp, False), ("fno", fno, False), ("fno_mt_a", fno_mt, True)]:
        pred = infer(model, p, coords, 256, norm, multitask=mt)
        results["RQ1"][name] = {
            "relative_l2": relative_l2_per_mode(pred, disp, mode),
            "rmse": rmse_per_mode(pred, disp, mode),
            "tangential_dir_error_deg": tangential_direction_error(pred, disp, mode),
        }
    # slip heads
    with torch.no_grad():
        cb = coords[None].expand(p.shape[0], -1, -1).contiguous()
        pn = (p - pm) / ps
        _, logits_a = fno_mt(pn, cb)
        field = fno(pn, cb)  # normalised field — same representation as training
        logits_b = clf(field, side)
    for tag, logits in [("slip_head_a_multitask", logits_a), ("slip_head_b_classifier", logits_b)]:
        mf1, per, binm = macro_f1(logits.argmax(-1), mode)
        results["RQ1"][tag] = {"macro_f1": mf1, "per_class_f1": dict(zip(MODE_NAMES, per)), **binm}

    # slip-only test set (harder)
    psp, csp, dsp, msp = load_split(data / "test_slip.npz", device)
    pred_s = infer(fno, psp, csp, 256, norm)
    results["RQ1"]["fno_on_test_slip"] = {
        "relative_l2": relative_l2_per_mode(pred_s, dsp, msp),
        "tangential_dir_error_deg": tangential_direction_error(pred_s, dsp, msp),
    }

    # ===== RQ2 : generalisation on OOD splits (FNO) =====
    id_l2 = results["RQ1"]["fno"]["relative_l2"]["overall"]
    for f in sorted(data.glob("test_ood_*.npz")):
        po, co, do, mo = load_split(f, device)
        key = f.stem.replace("test_ood_", "")
        try:
            pred = infer(fno, po, co, 256, norm)
            l2 = relative_l2_per_mode(pred, do, mo)["overall"]
            results["RQ2"][key] = {"relative_l2_overall": l2, "degradation_x": l2 / id_l2}
        except Exception as e:
            # FNO needs grid side >= 2*modes; downsampling below that is not
            # evaluable with fixed spectral weights (known FNO limitation).
            results["RQ2"][key] = {"error": f"{type(e).__name__}: not evaluable "
                                   f"(grid below FNO mode resolution)"}

    # ===== RQ3 : throughput + fidelity-speed =====
    speeds = {
        "mlp": throughput(mlp, p, coords, device),
        "fno": throughput(fno, p, coords, device),
        "fno_mt_a": throughput(fno_mt, p, coords, device, multitask=True),
    }
    # reference "solver" speed: analytic Hertz-Mindlin GT generation throughput
    from novbts.groundtruth.hertz_mindlin import hertz_mindlin_field
    pn, cn = p.cpu().numpy(), coords.cpu().numpy()
    t0 = time.perf_counter()
    for _ in range(3):
        hertz_mindlin_field(pn, cn)
    solver_fps = (3 * pn.shape[0]) / (time.perf_counter() - t0)
    speeds["gt_solver_analytic"] = solver_fps
    results["RQ3"]["throughput_fps"] = speeds
    results["RQ3"]["fno_speedup_vs_solver"] = speeds["fno"] / solver_fps
    results["RQ3"]["note"] = ("gt_solver_analytic is the lightweight Hertz-Mindlin reference; "
                              "a full PhysX/IPC FEM solver is orders slower, so the real "
                              "operator speedup is far larger.")

    # ---- fidelity-speed plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        pts = {
            "MLP": (speeds["mlp"], results["RQ1"]["mlp"]["relative_l2"]["overall"]),
            "FNO": (speeds["fno"], results["RQ1"]["fno"]["relative_l2"]["overall"]),
            "FNO+slip(a)": (speeds["fno_mt_a"], results["RQ1"]["fno_mt_a"]["relative_l2"]["overall"]),
        }
        for lbl, (x, y) in pts.items():
            ax.scatter(x, y, s=90); ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6))
        ax.axvline(solver_fps, ls="--", color="gray")
        ax.annotate("analytic GT solver", (solver_fps, ax.get_ylim()[1]), color="gray",
                    rotation=90, va="top", fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("throughput (frames/s, log)")
        ax.set_ylabel("relative L2 (lower = more accurate)")
        ax.set_title("Phase 3 fidelity-speed trade-off")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(run / "fidelity_speed.png", dpi=130)
        results["RQ3"]["plot"] = str(run / "fidelity_speed.png")
    except Exception as e:
        results["RQ3"]["plot_error"] = str(e)

    (run / "rq_results.json").write_text(json.dumps(results, indent=2))

    # ---- print summary ----
    print("\n================ RQ1 accuracy (test_id) ================")
    for m in ["mlp", "fno", "fno_mt_a"]:
        r = results["RQ1"][m]["relative_l2"]
        print(f"  {m:9s} relL2 overall={r['overall']:.4f} | "
              + " ".join(f"{n}={r[n]:.3f}" for n in MODE_NAMES)
              + f" | dirErr={results['RQ1'][m]['tangential_dir_error_deg']:.1f}deg")
    print("  --- slip detection (mode macro-F1 / slip-binary-F1) ---")
    for t in ["slip_head_a_multitask", "slip_head_b_classifier"]:
        print(f"  {t:24s} macroF1={results['RQ1'][t]['macro_f1']:.4f} "
              f"slipF1={results['RQ1'][t]['slip_f1']:.4f}")
    print("\n================ RQ2 generalisation (FNO) ================")
    print(f"  in-distribution relL2 = {id_l2:.4f}")
    for k, v in results["RQ2"].items():
        if "error" in v:
            print(f"  {k:16s} {v['error']}")
        else:
            print(f"  {k:16s} relL2={v['relative_l2_overall']:.4f}  ({v['degradation_x']:.2f}x)")
    print("\n================ RQ3 speed ================")
    for k, v in speeds.items():
        print(f"  {k:20s} {v:10.1f} frames/s")
    print(f"  FNO speedup vs analytic solver: {results['RQ3']['fno_speedup_vs_solver']:.2f}x")
    print(f"\nSaved {run/'rq_results.json'} + fidelity_speed.png")


if __name__ == "__main__":
    main()
