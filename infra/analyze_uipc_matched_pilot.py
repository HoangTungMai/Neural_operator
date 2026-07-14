#!/usr/bin/env python3
"""Analyze the exact-production-provenance UIPC matched pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def shape_metrics(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    c = np.asarray(coarse[:, :2], dtype=np.float64).reshape(-1)
    f = np.asarray(fine[:, :2], dtype=np.float64).reshape(-1)
    cosine = float(np.dot(c, f) / (np.linalg.norm(c) * np.linalg.norm(f) + 1e-30))
    alpha = float(np.dot(c, f) / (np.dot(c, c) + 1e-30))
    return {
        "raw_rel_l2": rel_l2(c, f),
        "cosine": cosine,
        "optimal_scale": alpha,
        "scaled_rel_l2": rel_l2(alpha * c, f),
    }


def scalar(data: np.lib.npyio.NpzFile, key: str, default: float = float("nan")) -> float:
    if key not in data.files:
        return default
    return float(np.asarray(data[key]).reshape(-1)[0])


def avg_path(root: Path, frame: int, nz: int, tol: str) -> Path:
    return (
        root / f"f{frame}" / f"nz{nz}_tol{tol}" /
        f"frame_{frame:03d}" / "uipc_gt_shear_avg.npz"
    )


def load_avg(root: Path, frame: int, nz: int, tol: str) -> dict[str, object]:
    path = avg_path(root, frame, nz, tol)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, object] = {
            "path": str(path),
            "disp": np.asarray(data["disp"][0], dtype=np.float64),
            "noise_tangential": scalar(data, "rep_noise_tangential"),
            "solve_time_sum_s": scalar(data, "solve_time_s"),
            "run_time_sum_s": scalar(data, "run_time_s"),
            "n_replicates": int(scalar(data, "raw_n_replicates", scalar(data, "n_replicates", 1))),
        }
        if "contact_force_total_normal" in data.files:
            out["force_normal"] = np.asarray(
                data["contact_force_total_normal"][0], dtype=np.float64
            )
        if "contact_force_total_friction" in data.files:
            out["force_friction"] = np.asarray(
                data["contact_force_total_friction"][0], dtype=np.float64
            )
    return out


def force_gap(a: dict[str, object], b: dict[str, object], key: str, component: str) -> float:
    if key not in a or key not in b:
        return float("nan")
    va = np.asarray(a[key], dtype=np.float64)
    vb = np.asarray(b[key], dtype=np.float64)
    if component == "z":
        return float(abs(va[2] - vb[2]) / (abs(vb[2]) + 1e-30))
    return float(abs(np.linalg.norm(va[:2]) - np.linalg.norm(vb[:2])) /
                 (np.linalg.norm(vb[:2]) + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--tol-anchor", default="0.0003")
    ap.add_argument("--tol-mesh", default="0.00001")
    ap.add_argument("--tol-check", default="0.000001")
    ap.add_argument("--old-wall-hours", type=float, default=19.556)
    ap.add_argument("--report", default="codex/REPORT_uipc_matched_pilot.md")
    args = ap.parse_args()

    root = Path(args.root)
    with np.load(args.data, allow_pickle=True) as production:
        prod_disp = np.asarray(production["disp"], dtype=np.float64)
        prod_mode = np.asarray(production["mode"], dtype=np.int32)

    frames: dict[str, object] = {}
    for frame in args.frames:
        anchor = load_avg(root, frame, 4, args.tol_anchor)
        mesh = {nz: load_avg(root, frame, nz, args.tol_mesh) for nz in (4, 8, 16, 24)}
        tight = load_avg(root, frame, 24, args.tol_check)
        successive = {}
        for a, b in ((4, 8), (8, 16), (16, 24)):
            successive[f"{a}_to_{b}"] = {
                "tangential": rel_l2(mesh[a]["disp"][:, :2], mesh[b]["disp"][:, :2]),
                "normal": rel_l2(mesh[a]["disp"][:, 2], mesh[b]["disp"][:, 2]),
            }
        frame_result = {
            "mode": int(prod_mode[frame]),
            "anchor_vs_production": {
                "tangential": rel_l2(anchor["disp"][:, :2], prod_disp[frame, :, :2]),
                "normal": rel_l2(anchor["disp"][:, 2], prod_disp[frame, :, 2]),
                "overall": rel_l2(anchor["disp"], prod_disp[frame]),
            },
            "mesh_4_to_24_shape": shape_metrics(mesh[4]["disp"], mesh[24]["disp"]),
            "mesh_successive": successive,
            "tol_1e5_to_1e6": {
                "tangential": rel_l2(mesh[24]["disp"][:, :2], tight["disp"][:, :2]),
                "normal": rel_l2(mesh[24]["disp"][:, 2], tight["disp"][:, 2]),
                "noise_tangential_1e5": mesh[24]["noise_tangential"],
                "noise_tangential_1e6": tight["noise_tangential"],
            },
            "force_4_to_24": {
                "normal_fz": force_gap(mesh[4], mesh[24], "force_normal", "z"),
                "friction_xy": force_gap(mesh[4], mesh[24], "force_friction", "xy"),
            },
            "timing": {
                f"nz{nz}_tol1e5": {
                    "solve_per_rep_s": mesh[nz]["solve_time_sum_s"] / mesh[nz]["n_replicates"],
                    "run_per_rep_s": mesh[nz]["run_time_sum_s"] / mesh[nz]["n_replicates"],
                }
                for nz in (4, 8, 16, 24)
            },
        }
        frame_result["timing"]["nz24_tol1e6"] = {
            "solve_per_rep_s": tight["solve_time_sum_s"] / tight["n_replicates"],
            "run_per_rep_s": tight["run_time_sum_s"] / tight["n_replicates"],
        }
        frames[str(frame)] = frame_result

    timing_path = root / "container_wall_times.tsv"
    container_rows = []
    if timing_path.exists():
        with timing_path.open(newline="") as handle:
            container_rows = list(csv.DictReader(handle, delimiter="\t"))

    tol_pass = all(
        frames[str(frame)]["tol_1e5_to_1e6"]["tangential"] < 0.01
        and frames[str(frame)]["tol_1e5_to_1e6"]["noise_tangential_1e5"] < 0.01
        and frames[str(frame)]["tol_1e5_to_1e6"]["noise_tangential_1e6"] < 0.01
        for frame in args.frames
    )
    shape_bad_count = sum(
        frames[str(frame)]["mesh_4_to_24_shape"]["scaled_rel_l2"] > 0.10
        for frame in args.frames
    )
    fine_increment = {
        str(frame): frames[str(frame)]["mesh_successive"]["16_to_24"]["tangential"]
        for frame in args.frames
    }
    fine_pass = all(value < 0.05 for value in fine_increment.values())
    mode_counts = np.bincount(prod_mode, minlength=4)
    nz24_run_by_mode = {
        int(frames[str(frame)]["mode"]):
            float(frames[str(frame)]["timing"]["nz24_tol1e5"]["run_per_rep_s"])
        for frame in args.frames
    }
    nonnormal_weight = int(mode_counts[1:].sum())
    nonnormal_mean_run_s = float(sum(
        int(mode_counts[mode]) * nz24_run_by_mode[mode] for mode in (1, 2, 3)
    ) / nonnormal_weight)
    # No normal frame was requested. Bound its cost by the min/max of the three
    # sampled non-normal modes instead of silently inventing a point estimate.
    known_weighted_sum = sum(
        int(mode_counts[mode]) * nz24_run_by_mode[mode] for mode in (1, 2, 3)
    )
    sampled_times = list(nz24_run_by_mode.values())
    mean_run_bounds_s = [
        float((known_weighted_sum + int(mode_counts[0]) * bound) / len(prod_mode))
        for bound in (min(sampled_times), max(sampled_times))
    ]
    projected_run_hours = [
        float(value * len(prod_mode) / 3600.0) for value in mean_run_bounds_s
    ]

    result = {
        "production_data": args.data,
        "root": str(root),
        "frames": frames,
        "gates": {
            "tol_pass_all_frames": tol_pass,
            "shape_scaled_gt10pct_count": shape_bad_count,
            "fine_16_to_24_lt5pct_all_frames": fine_pass,
            "fine_16_to_24": fine_increment,
        },
        "container_wall": {
            "records": container_rows,
            "sum_s": float(sum(float(row["container_wall_s"]) for row in container_rows)),
        },
        "cost_projection": {
            "production_mode_counts": mode_counts.tolist(),
            "nz24_tol1e5_run_s_by_sampled_mode": nz24_run_by_mode,
            "nonnormal_weighted_mean_run_s": nonnormal_mean_run_s,
            "mean_run_s_per_frame_bounds_normal_unsampled": mean_run_bounds_s,
            "run_hours_2520_bounds_excluding_boot_save": projected_run_hours,
            "old_campaign_wall_hours_lower_bound": args.old_wall_hours,
            "cheaper_than_old_even_before_boot_save": projected_run_hours[1] < args.old_wall_hours,
        },
    }
    metrics_path = root / "pilot_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))

    mode_names = ["normal", "stick", "partial slip", "full slip"]
    lines = [
        "# Matched UIPC pilot — strength 30000",
        "",
        f"Production source: `{args.data}`.",
        "",
        "## Gate summary",
        "",
        f"- tol `1e-5 -> 1e-6` and replicate-spread `<1%` on all frames: **{tol_pass}**",
        f"- after-scale `nz4 -> nz24 >10%`: **{shape_bad_count}/{len(args.frames)} frames**",
        f"- successive `nz16 -> nz24 <5%` on all frames: **{fine_pass}**",
        (f"- projected nz24/K1/tol1e-5 `run_one` cost: "
         f"**{projected_run_hours[0]:.1f}--{projected_run_hours[1]:.1f} h** "
         f"before boot/save (normal mode unsampled)"),
        "",
        "## Per-frame results",
        "",
        "| frame | mode | anchor→production uxy | nz4→24 raw | cosine | after scale | nz16→24 | tol 1e-5→1e-6 | noise 1e-5 / 1e-6 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frame in args.frames:
        row = frames[str(frame)]
        anchor = row["anchor_vs_production"]
        shape = row["mesh_4_to_24_shape"]
        fine = row["mesh_successive"]["16_to_24"]["tangential"]
        tol = row["tol_1e5_to_1e6"]
        lines.append(
            f"| {frame} | {mode_names[row['mode']]} | {anchor['tangential']:.2%} | "
            f"{shape['raw_rel_l2']:.2%} | {shape['cosine']:.4f} | "
            f"{shape['scaled_rel_l2']:.2%} | {fine:.2%} | {tol['tangential']:.2%} | "
            f"{tol['noise_tangential_1e5']:.2%} / {tol['noise_tangential_1e6']:.2%} |"
        )

    lines.extend([
        "",
        "## Timing (`run_one`, excludes boot/save)",
        "",
        "| frame | nz4 | nz8 | nz16 | nz24 @1e-5 | nz24 @1e-6 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for frame in args.frames:
        timing = frames[str(frame)]["timing"]
        lines.append(
            f"| {frame} | {timing['nz4_tol1e5']['run_per_rep_s']:.2f}s | "
            f"{timing['nz8_tol1e5']['run_per_rep_s']:.2f}s | "
            f"{timing['nz16_tol1e5']['run_per_rep_s']:.2f}s | "
            f"{timing['nz24_tol1e5']['run_per_rep_s']:.2f}s | "
            f"{timing['nz24_tol1e6']['run_per_rep_s']:.2f}s |"
        )
    lines.extend([
        "",
        f"Recorded container wall across executed/resumed settings: `{result['container_wall']['sum_s']:.1f}s`.",
        "",
        "## Decision logic",
        "",
    ])
    if not tol_pass:
        lines.append("**Do not launch `tol=1e-5` regen:** the tolerance/replicate gate failed.")
    elif shape_bad_count < 2:
        lines.append("**No full regen evidence:** fewer than two modes retain >10% shape error.")
    elif not fine_pass:
        lines.append(
            "**Refinement is warranted, but nz24 is not a demonstrated limit:** at least one "
            "`nz16 -> nz24` increment remains >=5%."
        )
    else:
        lines.append(
            "**Matched pilot scientifically supports regen at nz24/K1/tol1e-5, "
            "but refutes the claim that it is cheaper than the old campaign.**"
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(result["gates"], indent=2))
    print(f"Saved {metrics_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
