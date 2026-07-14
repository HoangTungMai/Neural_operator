#!/usr/bin/env python3
"""Prepare, validate, and analyze the production-matched GT convergence pilot."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FRAMES = (1362, 1812, 2508)
MODE_NAMES = ("normal", "stick", "partial slip", "full slip")
NUMERIC_META = {
    "gel_nz": 24,
    "gel_xy": 0.020,
    "gel_z": 0.003,
    "velocity_tol": 1e-5,
    "newton_max_iter": 1024,
    "gel_constraint_strength": 0.0,
    "indentor_constraint_strength": 30000.0,
    "marker_side": 32,
    "poisson": 0.45,
}
STRING_META = {"gel_bottom_bc": "fixed", "indentor_geom": "sphere"}


def scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data.files:
        raise AssertionError(f"missing provenance key {key}")
    return np.asarray(data[key]).reshape(-1)[0]


def read_configs(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: dict[str, dict[str, Any]] = {}
    integer = {
        "gel_res", "indentor_subdiv", "press_steps", "settle_steps",
        "shear_steps", "shear_settle",
    }
    for row in rows:
        cfg: dict[str, Any] = {"tag": row["tag"]}
        for key, value in row.items():
            if key == "tag":
                continue
            cfg[key] = int(value) if key in integer else float(value)
        out[row["tag"]] = cfg
    return out


def prepare_rows(source: Path, rows_path: Path, meta_path: Path) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=True) as data:
        params = np.asarray(data["params"], dtype=np.float64)
        modes = np.asarray(data["mode"], dtype=np.int32)
        load_modes = np.asarray(data["load_mode"], dtype=np.int32)
        load_names = str(np.asarray(data["load_mode_names"]).item()).split(",")
        rows: list[str] = []
        meta = ["frame\tmode\tmode_name\tdepth\tR\tmu\tE\tsx\tsy\tdrive_ratio\tload_mode"]
        for frame in FRAMES:
            p = params[frame]
            sx, sy, mu = float(p[4]), float(p[5]), float(p[6])
            g = float(np.hypot(sx, sy) / (max(mu, 1e-30) * 1e-3))
            load_mode = load_names[int(load_modes[frame])]
            rows.append(
                f"{frame} {p[2]:.17g} {g:.17g} {sx:.17g} {sy:.17g} "
                f"{load_mode} {p[3]:.17g} {mu:.17g} {p[7]:.17g}"
            )
            meta.append(
                f"{frame}\t{int(modes[frame])}\t{MODE_NAMES[int(modes[frame])]}\t"
                f"{p[2]:.17g}\t{p[3]:.17g}\t{mu:.17g}\t{p[7]:.17g}\t"
                f"{sx:.17g}\t{sy:.17g}\t{g:.17g}\t{load_mode}"
            )
    rows_path.write_text("\n".join(rows) + "\n")
    meta_path.write_text("\n".join(meta) + "\n")


def read_frame_rows(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        frame = int(t[0])
        out[frame] = {
            "depth": float(t[1]), "g": float(t[2]), "sx": float(t[3]),
            "sy": float(t[4]), "load_mode": t[5], "R": float(t[6]),
            "mu": float(t[7]), "youngs": float(t[8]),
        }
    return out


def result_path(root: Path, tag: str, frame: int) -> Path:
    return root / tag / f"frame_{frame:03d}" / "rep_1" / "uipc_gt_shear.npz"


def assert_close(path: Path, key: str, got: Any, want: Any) -> None:
    if isinstance(want, str):
        if str(got) != want:
            raise AssertionError(f"{path}: {key}={got!r}, expected {want!r}")
    elif not np.isclose(float(got), float(want), rtol=1e-6, atol=1e-12):
        raise AssertionError(f"{path}: {key}={got!r}, expected {want!r}")


def verify_setting(root: Path, cfg: dict[str, Any], frame_rows: dict[int, dict[str, Any]]) -> None:
    expected_cfg = {
        key: cfg[key]
        for key in (
            "gel_res", "indentor_subdiv", "d_hat", "eps_velocity",
            "contact_resistance", "press_steps", "settle_steps", "shear_steps",
            "shear_settle", "dt",
        )
    }
    expected_cfg.update(NUMERIC_META)
    expected_cfg.update(STRING_META)
    coords_ref: np.ndarray | None = None
    for frame in FRAMES:
        path = result_path(root, cfg["tag"], frame)
        if not path.exists():
            raise AssertionError(f"missing result {path}")
        try:
            with np.load(path, allow_pickle=True) as data:
                for key, want in expected_cfg.items():
                    assert_close(path, key, scalar(data, key), want)
                disp = np.asarray(data["disp"], dtype=np.float64)
                coords = np.asarray(data["coords"], dtype=np.float64)
                if disp.shape != (1, 1024, 3) or not np.isfinite(disp).all():
                    raise AssertionError(f"{path}: invalid disp shape/values {disp.shape}")
                if coords_ref is None:
                    coords_ref = coords
                elif not np.allclose(coords, coords_ref, rtol=0.0, atol=0.0):
                    raise AssertionError(f"{path}: marker coordinates differ")
                row = frame_rows[frame]
                params = np.asarray(data["params"], dtype=np.float64)[0]
                wants = (row["depth"], row["R"], row["sx"], row["sy"], row["mu"], row["youngs"])
                gets = (params[2], params[3], params[4], params[5], params[6], params[7])
                if not np.allclose(gets, wants, rtol=1e-6, atol=1e-10):
                    raise AssertionError(f"{path}: physical params {gets!r}, expected {wants!r}")
                load_names = str(np.asarray(data["load_mode_names"]).item()).split(",")
                load_mode = load_names[int(scalar(data, "load_mode"))]
                if load_mode != row["load_mode"]:
                    raise AssertionError(f"{path}: load_mode={load_mode}, expected {row['load_mode']}")
        except (OSError, ValueError) as exc:
            raise AssertionError(f"cannot load {path}: {exc}") from exc


def load_field(root: Path, tag: str, frame: int) -> tuple[np.ndarray, float]:
    with np.load(result_path(root, tag, frame), allow_pickle=True) as data:
        return (
            np.asarray(data["disp"], dtype=np.float64)[0, :, :2],
            float(np.asarray(data["run_time_s"]).reshape(-1)[0]),
        )


def field_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    ab = float(np.dot(a, b))
    amp = ab / (aa + 1e-30)  # candidate amplitude projected onto reference
    candidate_to_ref = ab / (bb + 1e-30)
    return {
        "raw_rel_l2": float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-30)),
        "amplitude_ratio": amp,
        "amplitude_change": abs(amp - 1.0),
        "shape_rel_l2": float(
            np.linalg.norm(candidate_to_ref * b - a) / (np.linalg.norm(a) + 1e-30)
        ),
        "cosine": ab / (math.sqrt(aa * bb) + 1e-30),
    }


def pair_metrics(root: Path, reference: str, candidate: str) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for frame in FRAMES:
        a, _ = load_field(root, reference, frame)
        b, _ = load_field(root, candidate, frame)
        frames[str(frame)] = field_metrics(a, b)
    return {"reference": reference, "candidate": candidate, "frames": frames}


def summarize(root: Path, configs_path: Path, rows_path: Path, output: Path) -> dict[str, Any]:
    configs = read_configs(configs_path)
    frame_rows = read_frame_rows(rows_path)
    available: dict[str, dict[str, Any]] = {}
    invalid: dict[str, str] = {}
    for tag, cfg in configs.items():
        try:
            verify_setting(root, cfg, frame_rows)
        except AssertionError as exc:
            invalid[tag] = str(exc)
            continue
        times = [load_field(root, tag, frame)[1] for frame in FRAMES]
        available[tag] = {
            "config": cfg,
            "run_time_s_by_frame": dict(zip(map(str, FRAMES), times)),
            "mean_run_time_s": float(np.mean(times)),
            "projected_2520_run_hours": float(np.mean(times) * 2520 / 3600),
        }

    requested_pairs = {
        "gel_res": ("prod", "gel_r32"),
        "indentor_subdiv": ("prod", "subdiv4"),
        "d_hat": ("prod", "dhat_5e5"),
        "eps_velocity": ("prod", "eps_1p25e5"),
        "contact_resistance": ("prod", "resistance_1e10"),
        "shear_steps": ("prod", "shear_160"),
        "dt_true_refinement": ("prod", "dt_half"),
    }
    pairs: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for knob, (a, b) in requested_pairs.items():
        if a not in available or b not in available:
            continue
        pair = pair_metrics(root, a, b)
        pairs[knob] = pair
        values = list(pair["frames"].values())
        ranking.append({
            "knob": knob,
            "comparison": f"{a}->{b}",
            "max_raw_rel_l2": max(v["raw_rel_l2"] for v in values),
            "max_amplitude_change": max(v["amplitude_change"] for v in values),
            "max_shape_rel_l2": max(v["shape_rel_l2"] for v in values),
            "median_raw_rel_l2": float(np.median([v["raw_rel_l2"] for v in values])),
        })
    ranking.sort(key=lambda row: row["max_raw_rel_l2"], reverse=True)

    sequences = {
        "d_hat": ["prod", "dhat_5e5", "dhat_2p5e5", "dhat_1p25e5", "dhat_6p25e6"],
        "eps_velocity": ["eps_5e5", "prod", "eps_1p25e5", "eps_6p25e6"],
        "contact_resistance": ["resistance_1e8", "prod", "resistance_1e10", "resistance_1e11"],
        "shear_steps": ["shear_40", "prod", "shear_160", "shear_320"],
        "indentor_subdiv": ["prod", "subdiv3", "subdiv4", "subdiv5"],
        "dt_true_refinement": ["prod", "dt_half", "dt_quarter"],
    }
    successive: dict[str, Any] = {}
    for axis, tags in sequences.items():
        if not all(tag in available for tag in tags):
            continue
        axis_rows = []
        for a, b in zip(tags[:-1], tags[1:]):
            axis_rows.append(pair_metrics(root, a, b))
        contractions: dict[str, list[float]] = {}
        for frame in FRAMES:
            increments = [row["frames"][str(frame)]["raw_rel_l2"] for row in axis_rows]
            contractions[str(frame)] = [
                increments[i + 1] / (increments[i] + 1e-30)
                for i in range(len(increments) - 1)
            ]
        successive[axis] = {"levels": tags, "increments": axis_rows, "contraction": contractions}

    richardson: dict[str, Any] = {}
    dhat_tags = sequences["d_hat"]
    if all(tag in available for tag in dhat_tags):
        for frame in FRAMES:
            fields = [load_field(root, tag, frame)[0] for tag in dhat_tags]
            diffs = [
                float(np.linalg.norm(fields[i + 1] - fields[i]))
                for i in range(len(fields) - 1)
            ]
            q = diffs[-2] / (diffs[-1] + 1e-30)
            row: dict[str, Any] = {"absolute_increment_norms": diffs, "last_contraction": 1.0 / q}
            if q > 1.0 + 1e-6:
                order = math.log(q, 2.0)
                limit = fields[-1] + (fields[-1] - fields[-2]) / (2.0**order - 1.0)
                row.update({"estimated_order": order, "last_to_extrapolated_limit": field_metrics(fields[-1], limit)})
            else:
                row["estimated_order"] = None
                row["last_to_extrapolated_limit"] = None
            richardson[str(frame)] = row

    g3: dict[str, Any] = {}
    if all(tag in available for tag in ("prod", "gel_r32", "subdiv4", "subdiv4_r32")):
        g3 = {
            "subdiv2_res24_to_32": pair_metrics(root, "prod", "gel_r32"),
            "subdiv4_res24_to_32": pair_metrics(root, "subdiv4", "subdiv4_r32"),
        }

    joint: dict[str, Any] = {}
    joint_pairs = {
        "production_to_budget": ("prod", "joint_budget"),
        "production_to_joint_fine": ("prod", "joint_fine"),
        "budget_to_temporal_converged": ("joint_budget", "joint_coarse"),
        "temporal_converged_to_joint_fine": ("joint_coarse", "joint_fine"),
        "budget_to_joint_fine": ("joint_budget", "joint_fine"),
    }
    for name, (reference, candidate) in joint_pairs.items():
        if reference in available and candidate in available:
            joint[name] = pair_metrics(root, reference, candidate)

    result = {
        "root": str(root),
        "configs": str(configs_path),
        "valid_settings": available,
        "invalid_or_missing_settings": invalid,
        "axis_pairs": pairs,
        "ranking": ranking,
        "successive_refinement": successive,
        "d_hat_richardson": richardson,
        "g3_indentor_mesh_interaction": g3,
        "joint_refinement": joint,
        "noise_interpretation_floor": 0.01,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--rows", type=Path, required=True)
    prep.add_argument("--meta", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--configs", type=Path, required=True)
    verify.add_argument("--rows", type=Path, required=True)
    verify.add_argument("--tag", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", type=Path, required=True)
    summary.add_argument("--configs", type=Path, required=True)
    summary.add_argument("--rows", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_rows(args.source, args.rows, args.meta)
        print(f"prepared {args.rows} and {args.meta}")
    elif args.command == "verify":
        configs = read_configs(args.configs)
        if args.tag not in configs:
            raise SystemExit(f"unknown tag {args.tag}")
        verify_setting(args.root, configs[args.tag], read_frame_rows(args.rows))
        print(f"verified {args.tag}: {len(FRAMES)} loadable NPZ files with matched provenance")
    else:
        result = summarize(args.root, args.configs, args.rows, args.output)
        print(json.dumps({
            "valid_settings": len(result["valid_settings"]),
            "invalid_or_missing_settings": len(result["invalid_or_missing_settings"]),
            "ranking": result["ranking"],
        }, indent=2))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
