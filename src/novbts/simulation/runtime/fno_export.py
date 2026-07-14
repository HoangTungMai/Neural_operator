#!/usr/bin/env python3
"""Export and load the frozen LR-FNO tactile surrogate for sim drivers.

The exported blob is deliberately plain tensors + JSON-like metadata so it can
be loaded by the host venv and the older torch in the Isaac container.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from novbts.research.fno.fem_benchmark import load as load_fem, norm_from
from novbts.research.fno.field2field import DEV, predict_raw, rel_l2_per_mode
from novbts.research.fno.hybrid_fno import make_field_model
from novbts.paths import ASSETS, RUNS, ensure


DEFAULT_CACHE = RUNS / "phase3_fem" / "geometry_ood_cache" / "lr_fno_seed0_ep80_lr0.001_m12.pt"
DEFAULT_DATA = Path("data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz")
DEFAULT_OUT = ASSETS / "vbts_fno" / "lr_fno_realistic_bc.pt"


def _to_cpu_norm(norm) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(x).detach().cpu() for x in norm)


def _coord_grid(side: int) -> torch.Tensor:
    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij"
    )[::-1], -1).astype(np.float32)
    return torch.tensor(grid, device=DEV)


def _param_box(params: np.ndarray) -> dict:
    return {
        "R_m": [float(params[:, 3].min()), float(params[:, 3].max())],
        "depth_m": [float(params[:, 2].min()), float(params[:, 2].max())],
        "shear_m": [float(np.hypot(params[:, 4], params[:, 5]).min()),
                    float(np.hypot(params[:, 4], params[:, 5]).max())],
        "mu": [float(params[:, 6].min()), float(params[:, 6].max())],
        "E_Pa": [float(params[:, 7].min()), float(params[:, 7].max())],
    }


def _load_training_data(data_path: str | Path) -> dict:
    data = load_fem(str(data_path))
    raw = np.load(data_path, allow_pickle=True)
    data["coords"] = np.asarray(raw["coords"], dtype=np.float32)
    data["params"] = np.asarray(raw["params"], dtype=np.float32)
    return data


@torch.no_grad()
def _tail_eval(model, data: dict, norm, n_test: int = 400) -> dict:
    side = int(data["side"])
    n = int(data["inp"].shape[0])
    nt = min(n_test, n)
    idx = torch.arange(n - nt, n, device=DEV)
    inp, out, scal, mode = (data[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    im, istd, om, ostd, sm, sstd = norm
    pred = predict_raw(
        model,
        (inp[idx] - im) / istd,
        (scal[idx] - sm) / sstd,
        _coord_grid(side),
        ostd,
        om,
    )
    return {
        "frames": int(nt),
        "relative_l2": rel_l2_per_mode(pred, out[idx], mode[idx]),
    }


def export_checkpoint(
    reuse_cache: str | Path = DEFAULT_CACHE,
    out: str | Path = DEFAULT_OUT,
    data_path: str | Path = DEFAULT_DATA,
    *,
    field_model: str = "lr_fno",
    modes: int = 12,
    n_test: int = 400,
) -> dict:
    cache_path = Path(reuse_cache)
    if not cache_path.exists():
        raise SystemExit(f"cache checkpoint not found: {cache_path}")
    data_path = Path(data_path)
    if not data_path.exists():
        raise SystemExit(f"training/eval data not found: {data_path}")

    blob = torch.load(cache_path, map_location=DEV, weights_only=False)
    state = blob["state_dict"]
    norm = tuple(torch.as_tensor(x, device=DEV) for x in blob["norm"])
    model = make_field_model(field_model, modes=modes).to(DEV)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    data = _load_training_data(data_path)
    eval_metrics = _tail_eval(model, data, norm, n_test=n_test)
    coords = np.asarray(data["coords"], dtype=np.float32)
    side = int(data["side"])
    meta = {
        "field_model": field_model,
        "modes": int(modes),
        "side": side,
        "coords_half_m": float(np.abs(coords).max()),
        "data_path": str(data_path),
        "source_cache": str(cache_path),
        "param_box": _param_box(np.asarray(data["params"], dtype=np.float32)),
        "source_train_s": float(blob.get("train_s", 0.0)),
        "source_peak_vram_gb": blob.get("peak_vram_gb"),
        "tail_eval": eval_metrics,
    }
    out = Path(out)
    ensure(out.parent)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in state.items()},
        "norm": _to_cpu_norm(norm),
        "coords": torch.tensor(coords, dtype=torch.float32),
        "side": side,
        "field_model": field_model,
        "modes": int(modes),
        "data_path": str(data_path),
        "param_box": meta["param_box"],
        "meta": meta,
    }, out)
    return {"out": str(out), "meta": meta}


def load_checkpoint(path: str | Path = DEFAULT_OUT, device: str | torch.device | None = None):
    """Return ``(model.eval() frozen, norm_on_device, meta)``."""
    dev = torch.device(device) if device is not None else DEV
    blob = torch.load(path, map_location=dev, weights_only=False)
    field_model = blob.get("field_model", blob.get("meta", {}).get("field_model", "lr_fno"))
    modes = int(blob.get("modes", blob.get("meta", {}).get("modes", 12)))
    model = make_field_model(field_model, modes=modes).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm = tuple(torch.as_tensor(x, device=dev) for x in blob["norm"])
    meta = dict(blob.get("meta", {}))
    meta.setdefault("side", int(blob.get("side", 32)))
    meta.setdefault("coords", blob.get("coords"))
    return model, norm, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--field-model", default="lr_fno", choices=["fno", "lr_fno"])
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=400)
    args = ap.parse_args()
    res = export_checkpoint(
        args.reuse_cache,
        args.out,
        args.data,
        field_model=args.field_model,
        modes=args.modes,
        n_test=args.n_test,
    )
    print(json.dumps(res, indent=2))
    print("FNO_EXPORT_OK", res["out"])


if __name__ == "__main__":
    main()
