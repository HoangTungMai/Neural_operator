#!/usr/bin/env python3
"""Geometry-OOD evaluation scaffold for IPC indenter-shape sweeps.

Tier-0/1 purpose: provide a runnable module that can gather provenance for each
OOD dataset and, when requested, train a sphere baseline once and evaluate it
zero-shot on geometry-shift datasets. Few-shot/scratch multi-seed reporting is
filled out in Phase E after all tiers have data.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.operator.fem_benchmark import load, norm_from
from novbts.operator.field2field import (
    DEV,
    predict_raw,
    rel_l2_per_mode,
    tangential_dir_error,
    train_operator,
)
from novbts.operator.hybrid_fno import make_field_model
from novbts.paths import RUNS, ensure


def _coord_grid(side: int) -> torch.Tensor:
    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij"
    )[::-1], -1).astype(np.float32)
    return torch.tensor(grid).to(DEV)


def _provenance(path: str, data: dict) -> dict:
    params = np.asarray(data["params"])
    geom_vals = sorted(set(float(x) for x in params[:, 8])) if params.shape[1] > 8 else []
    return {
        "gt": os.path.basename(path),
        "gt_path": path,
        "frames": int(params.shape[0]),
        "side": int(data["side"]),
        "geom_code_values": geom_vals,
        "radius_m": [float(params[:, 3].min()), float(params[:, 3].max())],
        "mu": [float(params[:, 6].min()), float(params[:, 6].max())],
        "E_Pa": [float(params[:, 7].min()), float(params[:, 7].max())],
        "meta": data["provenance"].get("meta", ""),
    }


def evaluate_geometry(model, flat_npz: str, norm=None, *, field_model: str = "lr_fno") -> dict:
    """Evaluate a trained field model on one flat IPC npz.

    Returns rel-L2 overall/per-mode plus tangential direction error. If ``model``
    is ``None`` this returns only dataset provenance, which keeps the Tier-0
    skeleton runnable before long baseline training exists.
    """
    data = load(flat_npz)
    out = {"provenance": _provenance(flat_npz, data)}
    if model is None:
        return out

    side = data["side"]
    cg = _coord_grid(side)
    inp, target, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    if norm is None:
        norm = norm_from(inp, target, scal)
    im, istd, om, ostd, sm, sstd = norm
    pred = predict_raw(model, (inp - im) / istd, (scal - sm) / sstd, cg, ostd, om)
    out.update({
        "field_model": field_model,
        "relative_l2": rel_l2_per_mode(pred, target, mode),
        "tangential_dir_error_deg": tangential_dir_error(pred, target, mode),
    })
    return out


def _eval_loaded(model, data: dict, norm, idx: torch.Tensor | None = None) -> dict:
    side = data["side"]
    cg = _coord_grid(side)
    inp, target, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    if idx is not None:
        inp, target, scal, mode = inp[idx], target[idx], scal[idx], mode[idx]
    im, istd, om, ostd, sm, sstd = norm
    pred = predict_raw(model, (inp - im) / istd, (scal - sm) / sstd, cg, ostd, om)
    return {
        "relative_l2": rel_l2_per_mode(pred, target, mode),
        "tangential_dir_error_deg": tangential_dir_error(pred, target, mode),
        "frames": int(inp.shape[0]),
    }


def adapt_geometry(model, flat_npz: str, norm, *, few_shot_n: int, test_n: int,
                   epochs: int, lr: float, modes: int, field_model: str,
                   scratch: bool = False) -> dict:
    """Fine-tune (or scratch-train) on a small OOD prefix and eval on a held-out suffix."""
    data = load(flat_npz)
    n = data["inp"].shape[0]
    if few_shot_n <= 0 or epochs <= 0:
        return {}
    if n <= few_shot_n:
        raise SystemExit(f"few_shot_n={few_shot_n} leaves no OOD test frames for {flat_npz} (N={n})")
    test_n = min(test_n, n - few_shot_n)
    tr = torch.arange(0, few_shot_n, device=DEV)
    te = torch.arange(n - test_n, n, device=DEV)
    inp, out, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    im, istd, om, ostd, sm, sstd = norm
    cg = _coord_grid(data["side"])
    if scratch:
        adapted = make_field_model(field_model, modes=modes).to(DEV)
    else:
        adapted = copy.deepcopy(model).to(DEV)
    secs, vram = train_operator(
        adapted,
        (inp[tr] - im) / istd,
        (out[tr] - om) / ostd,
        (scal[tr] - sm) / sstd,
        mode[tr],
        cg,
        epochs,
        lr,
    )
    metrics = _eval_loaded(adapted, data, norm, te)
    metrics.update({
        "train_frames": int(few_shot_n),
        "test_frames": int(test_n),
        "epochs": int(epochs),
        "lr": float(lr),
        "train_s": float(secs),
        "peak_vram_gb": vram,
    })
    return metrics


def _train_sphere_baseline(data_path: str, *, seed: int, n_test: int, epochs: int,
                           lr: float, modes: int, field_model: str):
    data = load(data_path)
    side, n = data["side"], data["inp"].shape[0]
    if n <= n_test:
        raise SystemExit(f"--n-test={n_test} leaves no train frames for {data_path} (N={n})")
    cg = _coord_grid(side)
    inp, out, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    tr = torch.arange(0, n - n_test, device=DEV)
    te = torch.arange(n - n_test, n, device=DEV)
    norm = norm_from(inp[tr], out[tr], scal[tr])
    im, istd, om, ostd, sm, sstd = norm

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_field_model(field_model, modes=modes).to(DEV)
    secs, vram = train_operator(
        model,
        (inp[tr] - im) / istd,
        (out[tr] - om) / ostd,
        (scal[tr] - sm) / sstd,
        mode[tr],
        cg,
        epochs,
        lr,
    )
    pred = predict_raw(model, (inp[te] - im) / istd, (scal[te] - sm) / sstd, cg, ostd, om)
    id_metrics = {
        "relative_l2": rel_l2_per_mode(pred, out[te], mode[te]),
        "tangential_dir_error_deg": tangential_dir_error(pred, out[te], mode[te]),
    }
    train_meta = _provenance(data_path, data)
    train_meta.update({"train_frames": int(n - n_test), "test_frames": int(n_test)})
    return model, norm, train_meta, id_metrics, {"train_s": float(secs), "peak_vram_gb": vram}


def _parse_dataset(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphere-train", default="data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz")
    ap.add_argument("--dataset", action="append", default=[],
                    help="OOD dataset as name=path; may be repeated")
    ap.add_argument("--out", default=str(RUNS / "phase3_fem" / "geometry_ood.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = provenance-only skeleton; >0 trains sphere baseline")
    ap.add_argument("--few-shot-n", type=int, default=0,
                    help="OOD frames used for transfer fine-tuning")
    ap.add_argument("--few-shot-epochs", type=int, default=0,
                    help="epochs for OOD few-shot fine-tuning")
    ap.add_argument("--few-shot-lr", type=float, default=1e-4)
    ap.add_argument("--ood-test-n", type=int, default=100)
    ap.add_argument("--scratch", action="store_true",
                    help="also train a same-budget scratch OOD model")
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--field-model", choices=["fno", "lr_fno"], default="lr_fno")
    args = ap.parse_args()

    datasets = _parse_dataset(args.dataset)
    summary = {
        "device": str(DEV),
        "field_model": args.field_model,
        "seed": args.seed,
        "mode_names": MODE_NAMES,
        "sphere_train": args.sphere_train,
        "datasets": {},
        "status": "provenance_only" if args.epochs <= 0 else "zero_shot",
    }

    model = None
    norm = None
    if args.epochs > 0:
        model, norm, train_meta, id_metrics, timing = _train_sphere_baseline(
            args.sphere_train,
            seed=args.seed,
            n_test=args.n_test,
            epochs=args.epochs,
            lr=args.lr,
            modes=args.modes,
            field_model=args.field_model,
        )
        summary["sphere_train_provenance"] = train_meta
        summary["in_distribution"] = id_metrics
        summary["training"] = timing
    else:
        sphere = load(args.sphere_train)
        summary["sphere_train_provenance"] = _provenance(args.sphere_train, sphere)

    for name, path in datasets.items():
        row = evaluate_geometry(model, path, norm, field_model=args.field_model)
        if model is not None and args.few_shot_n > 0 and args.few_shot_epochs > 0:
            row["few_shot"] = adapt_geometry(
                model,
                path,
                norm,
                few_shot_n=args.few_shot_n,
                test_n=args.ood_test_n,
                epochs=args.few_shot_epochs,
                lr=args.few_shot_lr,
                modes=args.modes,
                field_model=args.field_model,
            )
            if args.scratch:
                row["scratch"] = adapt_geometry(
                    model,
                    path,
                    norm,
                    few_shot_n=args.few_shot_n,
                    test_n=args.ood_test_n,
                    epochs=args.few_shot_epochs,
                    lr=args.few_shot_lr,
                    modes=args.modes,
                    field_model=args.field_model,
                    scratch=True,
                )
        summary["datasets"][name] = row

    out_path = Path(args.out)
    ensure(out_path.parent)
    out_path.write_text(json.dumps(summary, indent=2))
    print("GEOMETRY_OOD_DONE", out_path, flush=True)


if __name__ == "__main__":
    main()
