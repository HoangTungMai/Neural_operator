#!/usr/bin/env python3
"""Verify the realistic-geometry campaign and all downstream deliverables."""
from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/uipc/shear_res24_avg_swept_REALISTIC.npz"
GT_NAME = DATA.name
MODE_NAMES = ("normal", "stick", "partial", "full")
LEGACY_DATA = ROOT / "data/uipc/shear_res24_avg_swept.npz"
LEGACY_SHA256 = "c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def allclose_field(d: np.lib.npyio.NpzFile, key: str, value: float) -> None:
    require(key in d.files, f"dataset missing {key}")
    actual = np.asarray(d[key], dtype=np.float64).reshape(-1)
    require(np.allclose(actual, value), f"{key} is {np.unique(actual).tolist()}, expected {value}")


def verify_legacy_dataset() -> None:
    require(LEGACY_DATA.is_file(), f"legacy dataset was removed: {LEGACY_DATA.relative_to(ROOT)}")
    digest = hashlib.sha256(LEGACY_DATA.read_bytes()).hexdigest()
    require(digest == LEGACY_SHA256, "legacy shear_res24_avg_swept.npz was modified")
    print("LEGACY_DATASET_UNCHANGED", digest)


def verify_dataset() -> tuple[float, list[int]]:
    require(DATA.is_file(), f"missing {DATA.relative_to(ROOT)}")
    d = np.load(DATA, allow_pickle=True)
    for key in (
        "params", "coords", "disp", "mode", "solve_time_s", "n_replicates",
        "rep_noise_overall", "rep_noise_normal", "rep_noise_tangential",
        "gel_res", "eps_velocity", "velocity_tol", "d_hat",
        "contact_resistance", "marker_sampling",
        "split_test_size", "split_shuffle_seed", "source_frame_dir",
    ):
        require(key in d.files, f"dataset missing {key}")

    params = np.asarray(d["params"], dtype=np.float64)
    disp = np.asarray(d["disp"], dtype=np.float64)
    modes = np.asarray(d["mode"], dtype=np.int64).reshape(-1)
    solve_time = np.asarray(d["solve_time_s"], dtype=np.float64).reshape(-1)
    tang_noise = np.asarray(d["rep_noise_tangential"], dtype=np.float64).reshape(-1)
    require(params.shape == (2520, 9), f"params shape is {params.shape}")
    require(disp.shape == (2520, 1024, 3), f"disp shape is {disp.shape}")
    require(np.isfinite(params).all() and np.isfinite(disp).all(), "dataset contains NaN/Inf")
    require(np.isfinite(solve_time).all() and np.all(solve_time > 0), "invalid solve_time_s")
    require(np.isfinite(tang_noise).all() and np.all(tang_noise >= 0),
            "invalid rep_noise_tangential")

    counts = np.bincount(modes, minlength=4).tolist()
    require(set(np.unique(modes).tolist()) == {0, 1, 2, 3}, f"mode counts are {counts}")
    test_counts = np.bincount(modes[-400:], minlength=4).tolist()
    require(set(np.unique(modes[-400:]).tolist()) == {0, 1, 2, 3},
            f"test split does not contain all modes: {test_counts}")
    require(int(np.asarray(d["split_test_size"]).reshape(-1)[0]) == 400,
            "split_test_size is not 400")
    require(int(np.asarray(d["split_shuffle_seed"]).reshape(-1)[0]) == 2026,
            "split_shuffle_seed is not 2026")
    require(np.all(np.asarray(d["n_replicates"]).reshape(-1) == 3), "not every frame is K=3")
    nonnormal_noise = tang_noise[modes != 0]
    require(float(nonnormal_noise.mean()) <= 0.03,
            f"mean non-normal tangential replicate noise is {nonnormal_noise.mean():.2%}")
    require(np.all(np.asarray(d["gel_res"]).reshape(-1) == 24), "not every frame is res24")
    allclose_field(d, "eps_velocity", 2.5e-5)
    allclose_field(d, "velocity_tol", 1e-3)
    allclose_field(d, "d_hat", 1e-4)
    allclose_field(d, "contact_resistance", 1e9)
    sampling = np.asarray(d["marker_sampling"]).astype(str).reshape(-1)
    require(set(sampling.tolist()) == {"bilinear"}, "marker sampling is not uniformly bilinear")

    # Dataset fields are stored as float32; allow one-few ULP boundary roundoff.
    tol = 1e-7
    require(np.all((params[:, 2] >= 0.00015 - tol) & (params[:, 2] <= 0.00075 + tol)),
            "depth lies outside 0.15--0.75 mm")
    require(np.all((params[:, 3] >= 0.002 - tol) & (params[:, 3] <= 0.006 + tol)),
            "radius lies outside 2--6 mm")
    require(np.all((params[:, 6] >= 0.4 - tol) & (params[:, 6] <= 0.8 + tol)),
            "mu lies outside 0.4--0.8")
    require(np.all((params[:, 7] >= 0.5e5 - 1) & (params[:, 7] <= 2e5 + 1)),
            "Young's modulus lies outside the campaign box")
    require(np.all(params[:, 8] == 0), "non-sphere geometry found")
    g = np.linalg.norm(params[:, 4:6], axis=1) / (np.maximum(params[:, 6], 1e-12) * 0.001)
    require(np.all((g >= -tol) & (g <= 1.3 + tol)), "drive ratio lies outside 0--1.3")

    print("DATASET_OK", {
        "frames": len(modes),
        "modes": dict(zip(MODE_NAMES, counts)),
        "test_modes": dict(zip(MODE_NAMES, test_counts)),
        "mean_k3_solve_time_s": float(solve_time.mean()),
        "mean_single_solve_time_s": float(solve_time.mean() / 3.0),
        "mean_nonnormal_tangential_rep_noise": float(nonnormal_noise.mean()),
        "p95_nonnormal_tangential_rep_noise": float(np.quantile(nonnormal_noise, 0.95)),
    })
    return DATA.stat().st_mtime, counts


def verify_jsons(data_mtime: float) -> dict[str, dict]:
    paths = (
        "runs/phase3_fem/benchmark.json",
        "runs/phase3_fem/vbts_baselines.json",
        "runs/phase4/policy_servo.json",
        "runs/phase5/sensor_build.json",
        "runs/phase5/sensor_inverse_multiframe.json",
        "runs/phase6/env_demo.json",
    )
    loaded: dict[str, dict] = {}
    for rel in paths:
        path = ROOT / rel
        require(path.is_file(), f"missing {rel}")
        require(path.stat().st_mtime >= data_mtime, f"{rel} predates realistic dataset")
        obj = json.loads(path.read_text())
        require(obj.get("gt") == GT_NAME, f"{rel} gt={obj.get('gt')!r}")
        require(Path(obj.get("gt_path", "")).name == GT_NAME, f"{rel} has wrong gt_path")
        if rel == "runs/phase5/sensor_build.json":
            cosine = obj.get("flow_disp_cos_by_mode", {})
            require(set(cosine) == {"normal", "stick", "partial_slip", "full_slip"},
                    "sensor_build.json lacks per-mode flow/displacement cosine")
            for mode_name in ("stick", "partial_slip", "full_slip"):
                require(cosine[mode_name].get("mean") is not None,
                        f"sensor_build.json has no cosine for {mode_name}")
        if rel == "runs/phase3_fem/benchmark.json":
            r1 = obj.get("RQ1", {})
            fno_l2 = r1.get("fno", {}).get("relative_l2", {}).get("overall")
            mlp_l2 = r1.get("mlp", {}).get("relative_l2", {}).get("overall")
            require(fno_l2 is not None and mlp_l2 is not None and fno_l2 < mlp_l2,
                    "benchmark does not show FNO rel-L2 below MLP")
            fps = obj.get("RQ3", {}).get("throughput_fps", {})
            single = fps.get("gt_solver")
            k3 = fps.get("gt_solver_k3_averaged")
            require(single is not None and k3 is not None and single > k3,
                    "benchmark lacks distinct single-solve and K=3 solver rates")
            require(np.isclose(single / k3, 3.0, rtol=0.05),
                    "single-solve/K=3 solver-rate ratio is not approximately three")
        loaded[rel] = obj
    print("DOWNSTREAM_JSON_OK", list(paths))
    return loaded


def verify_figures(jsons: dict[str, dict]) -> None:
    deps = {
        "docs/kse2026/figs/fidelity_speed.png": "runs/phase3_fem/benchmark.json",
        "docs/kse2026/figs/policy_servo_curve.png": "runs/phase4/policy_servo.json",
        "docs/kse2026/figs/sensor_gt_vs_fno.png": "runs/phase5/sensor_inverse_multiframe.json",
    }
    for rel, dep in deps.items():
        path = ROOT / rel
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty {rel}")
        require(path.stat().st_mtime >= (ROOT / dep).stat().st_mtime, f"{rel} is stale")
    print("FIGURES_OK", list(deps))


def verify_paper() -> None:
    tex = ROOT / "docs/kse2026/main.tex"
    text = tex.read_text()
    stale = {
        "0.975": r"0\.975",
        "0.341": r"0\.341",
        "50x50": r"50\s*\\times\s*50",
        "490/430": r"490\s*/\s*430",
        "flat-punch": r"flat[- ]punch",
        "false 1.3% fidelity": r"1\.3\\?%.*confirming fidelity",
        "old FNO/MLP advantage": r"10\.6\s*\\times",
        "old baseline advantage low": r"4\.46",
        "old baseline advantage high": r"12\.69",
        "old control speedup": r"24\.2\s*\\times",
        "old autograd loss": r"2\.77\s*\\times\s*10\^\{-9\}",
        "old ES loss": r"1\.03\s*\\times\s*10\^\{-8\}",
        "old autograd wall": r"13\.67",
        "old ES wall": r"331\.19",
        "old ES target queries": r"\b704\b",
        "old oracle loss": r"1\.75\s*\\times\s*10\^\{-9\}",
        "overbroad sensor differentiability": r"whole sensor\s+differentiable end-to-end",
    }
    found = [name for name, pattern in stale.items() if re.search(pattern, text, re.IGNORECASE)]
    require(not found, f"paper contains stale claims: {found}")
    expected = {
        "new FNO/MLP advantage": r"12\.14\s*\\times",
        "new baseline advantage range": r"5\.54.*12\.63\s*\\times",
        "new target-query advantage": r"84\s*\\times",
        "new control wall speedup": r"24\.5\s*\\times",
        "new autograd loss": r"5\.70\s*\\times\s*10\^\{-10\}",
        "new ES loss": r"6\.21\s*\\times\s*10\^\{-10\}",
    }
    missing = [name for name, pattern in expected.items() if not re.search(pattern, text, re.IGNORECASE | re.DOTALL)]
    require(not missing, f"paper missing refreshed headline claims: {missing}")
    require(re.search(r"20\s*\\times\s*20\s*\\times\s*3", text) is not None,
            "paper does not state 20x20x3 mm geometry")

    pdf = ROOT / "docs/kse2026/main.pdf"
    log = ROOT / "docs/kse2026/main.log"
    require(pdf.is_file() and pdf.stat().st_mtime >= tex.stat().st_mtime, "paper PDF is missing/stale")
    require(log.is_file(), "paper build log is missing")
    log_text = log.read_text(errors="replace")
    require("undefined references" not in log_text.lower(), "LaTeX log has undefined references")
    require("undefined citation" not in log_text.lower(), "LaTeX log has undefined citations")
    print("PAPER_OK", {"pdf": str(pdf.relative_to(ROOT))})


def main() -> None:
    os.chdir(ROOT)
    verify_legacy_dataset()
    data_mtime, _ = verify_dataset()
    jsons = verify_jsons(data_mtime)
    verify_figures(jsons)
    verify_paper()
    print("REALISTIC_REGROUND_ACCEPTANCE_OK")


if __name__ == "__main__":
    main()
