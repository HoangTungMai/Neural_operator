#!/usr/bin/env python3
"""Verify corrected-BC UIPC reground dataset before Phase D."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novbts.groundtruth.uniform_shift_diagnostic import field_metrics  # noqa: E402


DATA = ROOT / "data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz"
LEGACY = {
    ROOT / "data/uipc/shear_res24_avg_swept.npz":
        "c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91",
    ROOT / "data/uipc/shear_res24_avg_swept_REALISTIC.npz":
        "c1f56ba6feb64abdd13f4fa31b4ef58f3af697e798c1e0e714478ea40e1ae68a",
}
MODE_NAMES = ("normal", "stick", "partial", "full")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_str(z: np.lib.npyio.NpzFile, key: str) -> set[str]:
    require(key in z.files, f"dataset missing {key}")
    return set(np.asarray(z[key]).astype(str).reshape(-1).tolist())


def allclose_key(z: np.lib.npyio.NpzFile, key: str, value: float, *, rtol=1e-5, atol=1e-8) -> None:
    require(key in z.files, f"dataset missing {key}")
    vals = np.asarray(z[key], dtype=np.float64).reshape(-1)
    require(np.allclose(vals, value, rtol=rtol, atol=atol),
            f"{key} has values {np.unique(vals).tolist()}, expected {value}")


def verify_legacy_sha() -> None:
    for path, expected in LEGACY.items():
        require(path.is_file(), f"legacy dataset missing: {path.relative_to(ROOT)}")
        digest = sha256(path)
        require(digest == expected, f"legacy dataset changed: {path.relative_to(ROOT)}")
        print("LEGACY_SHA_OK", path.relative_to(ROOT), digest)


def sampled_full_indices(modes: np.ndarray, max_samples: int = 200) -> np.ndarray:
    full = np.flatnonzero(modes == 3)
    require(full.size > 0, "dataset has no full-slip frames")
    if full.size <= max_samples:
        return full
    pick = np.linspace(0, full.size - 1, max_samples).round().astype(np.int64)
    return full[pick]


def verify_dataset() -> dict[str, object]:
    require(DATA.is_file(), f"missing {DATA.relative_to(ROOT)}")
    z = np.load(DATA, allow_pickle=True)
    for key in (
        "params", "coords", "disp", "mode", "solve_time_s", "n_replicates",
        "raw_n_replicates", "rep_noise_tangential", "rep_noise_tangential_raw",
        "gel_bottom_bc", "indentor_constraint_strength", "velocity_tol",
        "adaptive_tol", "adaptive_kept_count", "split_test_size",
        "split_shuffle_seed", "kept_replicate_indices", "rejected_replicate_indices",
    ):
        require(key in z.files, f"dataset missing {key}")

    params = np.asarray(z["params"], dtype=np.float64)
    coords = np.asarray(z["coords"], dtype=np.float64)
    disp = np.asarray(z["disp"], dtype=np.float64)
    modes = np.asarray(z["mode"], dtype=np.int64).reshape(-1)
    n = params.shape[0]
    require(2500 <= n <= 2540, f"unexpected frame count: {n}")
    require(params.shape == (n, 9), f"params shape is {params.shape}")
    require(disp.shape == (n, 1024, 3), f"disp shape is {disp.shape}")
    require(coords.shape[0] == 1024, f"coords shape is {coords.shape}")
    require(np.isfinite(params).all() and np.isfinite(disp).all(), "dataset contains NaN/Inf")

    require(unique_str(z, "gel_bottom_bc") == {"fixed"}, "gel_bottom_bc is not uniformly fixed")
    allclose_key(z, "indentor_constraint_strength", 30000.0)
    allclose_key(z, "velocity_tol", 0.0003)
    require("adaptive_tol" in z.files, "adaptive_tol missing")
    require(int(np.asarray(z["split_test_size"]).reshape(-1)[0]) == 400, "split_test_size is not 400")
    require(int(np.asarray(z["split_shuffle_seed"]).reshape(-1)[0]) == 2026,
            "split_shuffle_seed is not 2026")

    mode_counts = np.bincount(modes, minlength=4)
    require(set(np.unique(modes).tolist()) == {0, 1, 2, 3},
            f"mode counts are {dict(zip(MODE_NAMES, mode_counts.tolist()))}")

    keep = np.asarray(z["adaptive_kept_count"], dtype=np.int64).reshape(-1)
    raw = np.asarray(z["raw_n_replicates"], dtype=np.int64).reshape(-1)
    require(keep.shape == (n,), "adaptive_kept_count does not align with frames")
    require(raw.shape == (n,), "raw_n_replicates does not align with frames")
    require(np.all((keep >= 2) & (keep <= raw)), "adaptive kept count lies outside [2, raw]")
    require(np.all(raw == 5), f"raw_n_replicates is not uniformly 5: {np.unique(raw).tolist()}")

    sampled = sampled_full_indices(modes)
    uniform = np.array([
        field_metrics(disp[i], coords)["uniform_energy_fraction"]
        for i in sampled
    ], dtype=np.float64)
    require(float(uniform.mean()) < 0.40,
            f"full-slip sampled uniform mean is {uniform.mean():.2%}")

    print("DATASET_BC_OK", {
        "frames": int(n),
        "modes": {name: int(count) for name, count in zip(MODE_NAMES, mode_counts)},
        "uniform_full_mean": float(uniform.mean()),
        "uniform_full_p95": float(np.quantile(uniform, 0.95)),
        "uniform_full_max": float(uniform.max()),
    })

    print("ADAPTIVE_KEEP_DISTRIBUTION", {
        str(int(k)): int(v) for k, v in zip(*np.unique(keep, return_counts=True))
    })
    for mode_id, name in enumerate(MODE_NAMES):
        mask = modes == mode_id
        vals = keep[mask]
        noise = np.asarray(z["rep_noise_tangential"], dtype=np.float64).reshape(-1)[mask]
        print("MODE_SUMMARY", name, {
            "frames": int(mask.sum()),
            "kept": {str(int(k)): int(v) for k, v in zip(*np.unique(vals, return_counts=True))},
            "rep_noise_tangential_mean": float(noise.mean()) if noise.size else None,
            "rep_noise_tangential_p95": float(np.quantile(noise, 0.95)) if noise.size else None,
        })

    return {"frames": int(n)}


def main() -> None:
    verify_legacy_sha()
    verify_dataset()
    print("BC_REGROUND_ACCEPTANCE_OK")


if __name__ == "__main__":
    main()
