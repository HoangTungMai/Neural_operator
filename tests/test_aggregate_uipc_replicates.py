#!/usr/bin/env python3
"""Self-test for UIPC replicate aggregation contracts.

Run with:
  .venv-gate2/bin/python tests/test_aggregate_uipc_replicates.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novbts.groundtruth.aggregate_uipc_replicates import (  # noqa: E402
    aggregate_sweep,
    average_files,
    stratified_train_test_order,
)


def write_rep(path: Path, disp: np.ndarray, solve_s: float = 1.0) -> None:
    params = np.array([[0.0, 0.0, 3e-4, 4e-3, 2e-4, 0.0, 0.6, 1e5, 0.0]], dtype=np.float32)
    coords = np.zeros((4, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        params=params,
        coords=coords,
        disp=disp[None].astype(np.float32),
        mode=np.array([3], dtype=np.int32),
        solve_time_s=np.array([solve_s], dtype=np.float32),
        gel_bottom_bc=np.array(["fixed"]),
        indentor_constraint_strength=np.array([30000.0], dtype=np.float32),
    )


def make_reps(frame_dir: Path, n: int, *, outlier: bool) -> list[str]:
    base = np.array(
        [[1.0, 0.0, 0.2], [1.1, 0.0, 0.3], [0.9, 0.0, 0.2], [1.05, 0.0, 0.25]],
        dtype=np.float32,
    ) * 1e-4
    paths = []
    for i in range(n):
        rep = frame_dir / f"rep_{i + 1}"
        rep.mkdir(parents=True, exist_ok=True)
        field = base.copy()
        if outlier and i == n - 1:
            field[:, :2] *= 8.0
        else:
            field[:, :2] *= 1.0 + 0.005 * i
        path = rep / "uipc_gt_shear.npz"
        write_rep(path, field, solve_s=float(i + 1))
        paths.append(str(path))
    return paths


def test_adaptive_drop_outlier_and_contracts(tmp: Path) -> None:
    paths = make_reps(tmp / "frame", 5, outlier=True)
    avg = average_files(paths, adaptive_tol=0.05)

    assert avg["adaptive_kept_count"].tolist() == [4]
    assert avg["kept_replicate_indices"].tolist() == [1, 2, 3, 4]
    assert avg["rejected_replicate_indices"].tolist() == [5]
    for key in (
        "adaptive_tol",
        "adaptive_floor",
        "adaptive_pairwise_score",
        "robust_pairwise_rel_l2",
        "kept_replicate_indices",
        "rejected_replicate_indices",
    ):
        assert key in avg

    assert avg["solve_time_s"].tolist() == [15.0]
    assert avg["n_replicates"].tolist() == [4]
    assert avg["raw_n_replicates"].tolist() == [5]


def test_adaptive_floor_fallback_and_mixed_raw_k(tmp: Path) -> None:
    sweep = tmp / "sweep"
    make_reps(sweep / "combo_000" / "frame_000", 5, outlier=True)
    make_reps(sweep / "combo_000" / "frame_001", 4, outlier=True)

    out = tmp / "merged.npz"
    aggregate_sweep(
        str(sweep),
        str(out),
        mode_shear_scale=0.001,
        adaptive_tol=0.0,
        adaptive_floor=2,
        test_size=None,
    )
    z = np.load(out, allow_pickle=True)
    assert z["adaptive_kept_count"].tolist() == [2, 2]
    assert z["raw_n_replicates"].tolist() == [5, 4]
    assert z["robust_pairwise_rel_l2"].shape == (2, 5, 5)


def test_stratified_order_is_deterministic() -> None:
    modes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    a = stratified_train_test_order(modes, test_size=4, seed=2026)
    b = stratified_train_test_order(modes, test_size=4, seed=2026)
    c = stratified_train_test_order(modes, test_size=4, seed=2027)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(modes[a[-4:]]).tolist()) == {0, 1, 2, 3}


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="uipc_agg_test_"))
    try:
        test_adaptive_drop_outlier_and_contracts(tmp / "case1")
        test_adaptive_floor_fallback_and_mixed_raw_k(tmp / "case2")
        test_stratified_order_is_deterministic()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AGGREGATE_UIPC_REPLICATES_TEST_PASS")


if __name__ == "__main__":
    main()
