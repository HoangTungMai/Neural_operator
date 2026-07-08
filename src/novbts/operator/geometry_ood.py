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
import time
from pathlib import Path

import numpy as np
import torch

from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.operator.fem_benchmark import load, norm_from
from novbts.operator.field2field import (
    DEV,
    params_to_fieldinput,
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


def _default_datasets() -> dict[str, str]:
    return {
        "cylinder": "data/uipc/geom_ood/cylinder/cylinder_avg.npz",
        "sphere_oodR": "data/uipc/geom_ood/sphere_oodR/sphere_oodR_avg.npz",
        "cuboid": "data/uipc/geom_ood/cuboid/cuboid_avg.npz",
        "ellipsoid": "data/uipc/geom_ood/ellipsoid/ellipsoid_avg.npz",
        "bolt_hex": "data/uipc/geom_ood/mesh/bolt_hex/bolt_hex_avg.npz",
        "rounded_tip": "data/uipc/geom_ood/mesh/rounded_tip/rounded_tip_avg.npz",
    }


def _load(npz: str) -> dict:
    data = load(npz)
    raw = np.load(npz, allow_pickle=True)
    data["coords"] = np.asarray(raw["coords"], dtype=np.float32)
    if "contact_profile" in raw.files:
        data["contact_profile"] = np.asarray(raw["contact_profile"], dtype=np.float32)
    return data


def _provenance(path: str, data: dict) -> dict:
    params = np.asarray(data["params"])
    geom_vals = sorted(set(float(x) for x in params[:, 8])) if params.shape[1] > 8 else []
    return {
        "gt": os.path.basename(path),
        "gt_path": path,
        "frames": int(params.shape[0]),
        "side": int(data["side"]),
        "geom_code_values": geom_vals,
        "params_cols": int(params.shape[1]),
        "radius_m": [float(params[:, 3].min()), float(params[:, 3].max())],
        "radius2_m": (
            [float(params[:, 9].min()), float(params[:, 9].max())]
            if params.shape[1] > 9 else None
        ),
        "mu": [float(params[:, 6].min()), float(params[:, 6].max())],
        "E_Pa": [float(params[:, 7].min()), float(params[:, 7].max())],
        "meta": data["provenance"].get("meta", ""),
    }


def _rmse(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.square())).detach().cpu() * 1.0e6)


def abs_rmse_um(pred: torch.Tensor, tgt: torch.Tensor, mode: torch.Tensor) -> dict:
    """Absolute displacement RMSE in micrometres, split by channel and mode."""
    err = pred - tgt

    def pack(e: torch.Tensor) -> dict[str, float]:
        return {
            "all": _rmse(e),
            "normal": _rmse(e[:, 2:3]),
            "tangential": _rmse(e[:, :2]),
        }

    out = {"overall": pack(err)}
    for i, name in enumerate(MODE_NAMES):
        m = mode == i
        out[name] = pack(err[m]) if bool(m.any()) else {
            "all": float("nan"), "normal": float("nan"), "tangential": float("nan")
        }
    return out


def _geometry_distance(data: dict, train_radius_range=(0.002, 0.006)) -> float:
    """Geometry-shift scalar for the Phase-E x-axis.

    Base term: mean relative penetration-field shift vs the nearest in-range
    training sphere. This handles sphere_oodR by clamping R to the train range.
    Ellipsoid adds an anisotropy penalty, 2.5 * (R/R2 - 1), because the narrow
    axis is the main geometry shift but the binary footprint overlap alone
    under-ranks it versus edge/corner flat punches.
    """
    params = np.asarray(data["params"], dtype=np.float32)
    coords = np.asarray(data["coords"], dtype=np.float32)
    side = int(data["side"])
    ref = params.copy()
    ref[:, 3] = np.clip(ref[:, 3], train_radius_range[0], train_radius_range[1])
    ref[:, 8] = 0.0
    if ref.shape[1] > 9:
        ref[:, 9] = ref[:, 3]
    if "contact_profile" in data:
        cp = np.asarray(data["contact_profile"], dtype=np.float32)
        if cp.ndim == 2:
            cp = cp[None, ...]
        inp0 = cp.reshape(cp.shape[0], side, side)
    else:
        inp, _ = params_to_fieldinput(params, coords, side)
        inp0 = inp[:, 0]
    ref_inp, _ = params_to_fieldinput(ref, coords, side)
    num = np.linalg.norm((inp0 - ref_inp[:, 0]).reshape(inp0.shape[0], -1), axis=1)
    den = np.linalg.norm(ref_inp[:, 0].reshape(inp0.shape[0], -1), axis=1) + 1e-12
    base = float(np.mean(num / den))
    if params.shape[1] > 9 and np.any(np.abs(params[:, 8] - 5.0) < 0.5):
        anis = np.maximum(params[:, 3] / np.maximum(params[:, 9], 1e-12) - 1.0, 0.0)
        base += 2.5 * float(np.mean(anis))
    return base


def evaluate_geometry(model, flat_npz: str, norm=None, *, field_model: str = "lr_fno") -> dict:
    """Evaluate a trained field model on one flat IPC npz.

    Returns rel-L2 overall/per-mode plus tangential direction error. If ``model``
    is ``None`` this returns only dataset provenance, which keeps the Tier-0
    skeleton runnable before long baseline training exists.
    """
    data = _load(flat_npz)
    out = {"provenance": _provenance(flat_npz, data)}
    if model is None:
        return out

    inp, target, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    if norm is None:
        norm = norm_from(inp, target, scal)
    pred = _predict_loaded(model, data, norm)
    out.update({
        "field_model": field_model,
        "relative_l2": rel_l2_per_mode(pred, target, mode),
        "abs_rmse_um": abs_rmse_um(pred, target, mode),
        "tangential_dir_error_deg": tangential_dir_error(pred, target, mode),
        "geometry_distance": _geometry_distance(data),
    })
    return out


def _predict_loaded(model, data: dict, norm, idx: torch.Tensor | None = None) -> torch.Tensor:
    side = data["side"]
    cg = _coord_grid(side)
    inp, _target, scal = (data[k].to(DEV) for k in ("inp", "out", "scal"))
    if idx is not None:
        inp, scal = inp[idx], scal[idx]
    im, istd, om, ostd, sm, sstd = norm
    return predict_raw(model, (inp - im) / istd, (scal - sm) / sstd, cg, ostd, om)


def _eval_loaded(model, data: dict, norm, idx: torch.Tensor | None = None) -> dict:
    target, mode = (data[k].to(DEV) for k in ("out", "mode"))
    if idx is not None:
        target, mode = target[idx], mode[idx]
    pred = _predict_loaded(model, data, norm, idx)
    return {
        "relative_l2": rel_l2_per_mode(pred, target, mode),
        "abs_rmse_um": abs_rmse_um(pred, target, mode),
        "tangential_dir_error_deg": tangential_dir_error(pred, target, mode),
        "frames": int(target.shape[0]),
    }


def adapt_geometry(model, flat_npz: str, norm, *, few_shot_n: int, test_n: int,
                   epochs: int, lr: float, modes: int, field_model: str,
                   scratch: bool = False, seed: int | None = None) -> dict:
    """Fine-tune (or scratch-train) on a small OOD prefix and eval on a held-out suffix."""
    data = _load(flat_npz)
    n = data["inp"].shape[0]
    if few_shot_n <= 0 or epochs <= 0:
        return {}
    if n <= few_shot_n:
        raise SystemExit(f"few_shot_n={few_shot_n} leaves no OOD test frames for {flat_npz} (N={n})")
    test_n = min(test_n, n - few_shot_n)
    tr = torch.arange(0, few_shot_n, device=DEV)
    te = torch.arange(n - test_n, n, device=DEV)
    inp, out, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    cg = _coord_grid(data["side"])
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if scratch:
        adapted = make_field_model(field_model, modes=modes).to(DEV)
        train_norm = norm_from(inp[tr], out[tr], scal[tr])
    else:
        adapted = copy.deepcopy(model).to(DEV)
        train_norm = norm
    im, istd, om, ostd, sm, sstd = train_norm
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
    metrics = _eval_loaded(adapted, data, train_norm, te)
    metrics.update({
        "train_frames": int(few_shot_n),
        "test_frames": int(test_n),
        "epochs": int(epochs),
        "lr": float(lr),
        "train_s": float(secs),
        "peak_vram_gb": vram,
    })
    return metrics


def _norm_to_device(norm):
    return tuple(x.to(DEV) for x in norm)


def _train_sphere_baseline(data_path: str, *, seed: int, n_test: int, epochs: int,
                           lr: float, modes: int, field_model: str,
                           cache_dir: str | None = None):
    data = _load(data_path)
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
    cache_path = None
    cache_hit = False
    secs, vram = 0.0, None
    if cache_dir:
        cache = Path(cache_dir)
        ensure(cache)
        cache_path = cache / f"{field_model}_seed{seed}_ep{epochs}_lr{lr:g}_m{modes}.pt"
        if cache_path.exists():
            blob = torch.load(cache_path, map_location=DEV)
            model.load_state_dict(blob["state_dict"])
            norm = _norm_to_device(tuple(blob["norm"]))
            im, istd, om, ostd, sm, sstd = norm
            secs = float(blob.get("train_s", 0.0))
            vram = blob.get("peak_vram_gb")
            cache_hit = True
    if not cache_hit:
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
        if cache_path is not None:
            torch.save({
                "state_dict": model.state_dict(),
                "norm": tuple(x.detach().cpu() for x in norm),
                "train_s": float(secs),
                "peak_vram_gb": vram,
            }, cache_path)
    pred = predict_raw(model, (inp[te] - im) / istd, (scal[te] - sm) / sstd, cg, ostd, om)
    id_metrics = {
        "relative_l2": rel_l2_per_mode(pred, out[te], mode[te]),
        "abs_rmse_um": abs_rmse_um(pred, out[te], mode[te]),
        "tangential_dir_error_deg": tangential_dir_error(pred, out[te], mode[te]),
    }
    train_meta = _provenance(data_path, data)
    train_meta.update({"train_frames": int(n - n_test), "test_frames": int(n_test)})
    return model, norm, train_meta, id_metrics, {
        "train_s": float(secs),
        "peak_vram_gb": vram,
        "cache_hit": cache_hit,
        "cache_path": str(cache_path) if cache_path is not None else None,
    }


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


def _add_degradation(metrics: dict, id_metrics: dict) -> None:
    id_rel = id_metrics["relative_l2"]["overall"]
    id_tang = id_metrics["abs_rmse_um"]["overall"]["tangential"]
    id_norm = id_metrics["abs_rmse_um"]["overall"]["normal"]
    metrics["degradation_x"] = {
        "rel_l2_overall": float(metrics["relative_l2"]["overall"] / max(id_rel, 1e-12)),
        "tangential_abs_rmse": float(
            metrics["abs_rmse_um"]["overall"]["tangential"] / max(id_tang, 1e-12)
        ),
        "normal_abs_rmse": float(
            metrics["abs_rmse_um"]["overall"]["normal"] / max(id_norm, 1e-12)
        ),
    }


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def _recursive_stats(items: list):
    vals = [x for x in items if _is_number(x)]
    if len(vals) == len(items) and vals:
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}
    if all(isinstance(x, dict) for x in items):
        keys = sorted(set.intersection(*(set(x.keys()) for x in items)))
        out = {}
        for k in keys:
            v = _recursive_stats([x[k] for x in items])
            if v is not None:
                out[k] = v
        return out
    return None


def _aggregate_phase_e_model(runs: list[dict]) -> dict:
    out = {"in_distribution": _recursive_stats([r["in_distribution"] for r in runs])}
    names = sorted(runs[0]["datasets"].keys())
    out["datasets"] = {}
    for name in names:
        out["datasets"][name] = {
            "geometry_distance": _recursive_stats([
                r["datasets"][name]["geometry_distance"] for r in runs
            ]),
            "zero_shot": _recursive_stats([r["datasets"][name]["zero_shot"] for r in runs]),
            "few_shot": _recursive_stats([r["datasets"][name]["few_shot"] for r in runs]),
            "scratch": _recursive_stats([r["datasets"][name]["scratch"] for r in runs]),
        }
    return out


def _make_phase_e_figure(summary: dict, fig_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        summary["figure_error"] = f"{type(e).__name__}: {e}"
        return

    model_key = "lr_fno" if "lr_fno" in summary["models"] else next(iter(summary["models"]))
    ds = summary["models"][model_key]["aggregate"]["datasets"]
    rows = []
    for name, row in ds.items():
        x = row["geometry_distance"]["mean"]
        z = row["zero_shot"]["abs_rmse_um"]["overall"]["tangential"]
        f = row["few_shot"]["abs_rmse_um"]["overall"]["tangential"]
        rows.append((x, name, z["mean"], z["std"], f["mean"], f["std"]))
    rows.sort(key=lambda r: r[0])
    x = np.array([r[0] for r in rows], dtype=np.float64)
    labels = [r[1] for r in rows]
    z_mean = np.array([r[2] for r in rows], dtype=np.float64)
    z_std = np.array([r[3] for r in rows], dtype=np.float64)
    f_mean = np.array([r[4] for r in rows], dtype=np.float64)
    f_std = np.array([r[5] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(x, z_mean, yerr=z_std, marker="o", capsize=3, label="zero-shot")
    ax.errorbar(x, f_mean, yerr=f_std, marker="s", capsize=3, label="few-shot")
    offsets = {
        "sphere_oodR": (5, 6),
        "cylinder": (5, 8),
        "cuboid": (-18, 12),
        "ellipsoid": (6, 18),
    }
    for xi, yi, label in zip(x, z_mean, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=offsets.get(label, (5, 6)), fontsize=8)
    ax.set_xlabel("geometry distance (contact-shift + anisotropy)")
    ax.set_ylabel("tangential abs-RMSE (um)")
    ax.set_ylim(top=float(max(z_mean + z_std) * 1.18))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure(Path(fig_path).parent)
    fig.savefig(fig_path, dpi=160)
    summary["figure"] = fig_path
    summary["figure_model"] = model_key


def run_phase_e(args) -> dict:
    datasets = _parse_dataset(args.dataset) if args.dataset else _default_datasets()
    field_models = args.field_models or ["lr_fno", "fno"]
    baseline_epochs = args.epochs if args.epochs > 0 else 80
    summary = {
        "status": "phase_e",
        "scope": "6_shapes_with_tier4_bolt_hex_and_rounded_tip",
        "device": str(DEV),
        "mode_names": MODE_NAMES,
        "sphere_train": args.sphere_train,
        "seeds": args.seeds,
        "epochs": baseline_epochs,
        "few_shot": {
            "frames": args.few_shot_n,
            "epochs": args.few_shot_epochs,
            "lr": args.few_shot_lr,
            "ood_test_n": args.ood_test_n,
        },
        "metric_note": (
            "Figure uses tangential abs-RMSE in micrometres because rel-L2 is inflated "
            "when target norms differ across geometry footprints; rel-L2 is retained in JSON."
        ),
        "gt": os.path.basename(args.sphere_train),
        "gt_path": args.sphere_train,
        "datasets": {},
        "models": {},
    }
    sphere = _load(args.sphere_train)
    summary["sphere_train_provenance"] = _provenance(args.sphere_train, sphere)
    loaded_ood = {name: _load(path) for name, path in datasets.items()}
    for name, data in loaded_ood.items():
        summary["datasets"][name] = _provenance(datasets[name], data)
        summary["datasets"][name]["geometry_distance"] = _geometry_distance(data)

    for field_model in field_models:
        model_rows = []
        for seed in args.seeds:
            print(f"[PhaseE] model={field_model} seed={seed} train sphere", flush=True)
            t0 = time.perf_counter()
            model, norm, train_meta, id_metrics, timing = _train_sphere_baseline(
                args.sphere_train,
                seed=seed,
                n_test=args.n_test,
                epochs=baseline_epochs,
                lr=args.lr,
                modes=args.modes,
                field_model=field_model,
                cache_dir=args.cache_dir,
            )
            seed_row = {
                "seed": seed,
                "sphere_train_provenance": train_meta,
                "training": timing,
                "in_distribution": id_metrics,
                "datasets": {},
            }
            for offset, (name, path) in enumerate(datasets.items()):
                print(f"[PhaseE] model={field_model} seed={seed} eval {name}", flush=True)
                data = loaded_ood[name]
                zero = _eval_loaded(model, data, norm)
                zero["geometry_distance"] = _geometry_distance(data)
                _add_degradation(zero, id_metrics)
                few = adapt_geometry(
                    model,
                    path,
                    norm,
                    few_shot_n=args.few_shot_n,
                    test_n=args.ood_test_n,
                    epochs=args.few_shot_epochs,
                    lr=args.few_shot_lr,
                    modes=args.modes,
                    field_model=field_model,
                    seed=seed * 100 + offset,
                )
                _add_degradation(few, id_metrics)
                scratch = adapt_geometry(
                    model,
                    path,
                    norm,
                    few_shot_n=args.few_shot_n,
                    test_n=args.ood_test_n,
                    epochs=args.few_shot_epochs,
                    lr=args.few_shot_lr,
                    modes=args.modes,
                    field_model=field_model,
                    scratch=True,
                    seed=seed * 100 + offset + 10000,
                )
                _add_degradation(scratch, id_metrics)
                seed_row["datasets"][name] = {
                    "provenance": summary["datasets"][name],
                    "geometry_distance": zero["geometry_distance"],
                    "zero_shot": zero,
                    "few_shot": few,
                    "scratch": scratch,
                }
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
            seed_row["wall_s"] = time.perf_counter() - t0
            model_rows.append(seed_row)
            del model
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        summary["models"][field_model] = {
            "runs": model_rows,
            "aggregate": _aggregate_phase_e_model(model_rows),
        }

    _make_phase_e_figure(summary, args.figure)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphere-train", default="data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz")
    ap.add_argument("--dataset", action="append", default=[],
                    help="OOD dataset as name=path; may be repeated")
    ap.add_argument("--out", default=str(RUNS / "phase3_fem" / "geometry_ood_phaseE.json"))
    ap.add_argument("--phase-e", action="store_true",
                    help="run the full Phase-E 4-shape multi-seed matrix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=0,
                    help="sphere baseline epochs; Phase-E uses 80 when left at 0")
    ap.add_argument("--few-shot-n", type=int, default=50,
                    help="OOD frames used for transfer fine-tuning")
    ap.add_argument("--few-shot-epochs", type=int, default=20,
                    help="epochs for OOD few-shot fine-tuning")
    ap.add_argument("--few-shot-lr", type=float, default=1e-4)
    ap.add_argument("--ood-test-n", type=int, default=100)
    ap.add_argument("--scratch", action="store_true",
                    help="also train a same-budget scratch OOD model")
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--field-model", choices=["fno", "lr_fno"], default="lr_fno")
    ap.add_argument("--field-models", nargs="+", choices=["fno", "lr_fno"], default=None)
    ap.add_argument("--cache-dir", default=str(RUNS / "phase3_fem" / "geometry_ood_cache"))
    ap.add_argument("--figure", default="docs/kse2026/figs/geometry_ood.png")
    args = ap.parse_args()

    if args.phase_e:
        summary = run_phase_e(args)
        out_path = Path(args.out)
        ensure(out_path.parent)
        out_path.write_text(json.dumps(summary, indent=2))
        print("GEOMETRY_OOD_DONE", out_path, flush=True)
        return

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
            cache_dir=args.cache_dir,
        )
        summary["sphere_train_provenance"] = train_meta
        summary["in_distribution"] = id_metrics
        summary["training"] = timing
    else:
        sphere = _load(args.sphere_train)
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
