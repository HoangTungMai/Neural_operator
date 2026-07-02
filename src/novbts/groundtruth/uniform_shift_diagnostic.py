#!/usr/bin/env python3
"""Diagnose rigid/uniform tangential shift in IPC/UIPC marker fields."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _as_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-2], x.shape[-1])
    if x.ndim == 2:
        return x[None, ...]
    raise ValueError(f"expected disp [N,M,3] or [M,3], got shape {x.shape}")


def field_metrics(field: np.ndarray, coords: np.ndarray) -> dict:
    """Return rigid-shift diagnostics for one [M,3] marker field."""
    u = np.asarray(field[:, :2], dtype=np.float64) * 1e3
    mean = u.mean(axis=0)
    uniform = np.broadcast_to(mean, u.shape)
    residual = u - mean
    total_norm = np.linalg.norm(u.reshape(-1)) + 1e-12
    uniform_norm = np.linalg.norm(uniform.reshape(-1))
    residual_norm = np.linalg.norm(residual.reshape(-1))
    residual_mag = np.linalg.norm(residual, axis=1)
    tang_mag = np.linalg.norm(u, axis=1)

    depth = -np.asarray(field[:, 2], dtype=np.float64) * 1e3
    rr = np.hypot(coords[:, 0] - coords[:, 0].mean(), coords[:, 1] - coords[:, 1].mean())
    center = rr < np.percentile(rr, 10)
    edge = rr > np.percentile(rr, 75)

    return {
        "mean_shift_x_mm": float(mean[0]),
        "mean_shift_y_mm": float(mean[1]),
        "mean_shift_mag_mm": float(np.linalg.norm(mean)),
        "total_tangential_rms_mm": float(np.sqrt(np.mean(np.sum(u * u, axis=1)))),
        "uniform_norm_fraction": float(uniform_norm / total_norm),
        "uniform_energy_fraction": float((uniform_norm * uniform_norm) / (total_norm * total_norm)),
        "residual_norm_fraction": float(residual_norm / total_norm),
        "residual_tangential_rms_mm": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "residual_tangential_mean_mm": float(residual_mag.mean()),
        "residual_tangential_p95_mm": float(np.percentile(residual_mag, 95)),
        "residual_tangential_max_mm": float(residual_mag.max()),
        "max_tangential_mm": float(tang_mag.max()),
        "depth_p95_mm": float(np.percentile(depth, 95)),
        "depth_max_mm": float(depth.max()),
        "depth_range_mm": float(depth.max() - depth.min()),
        "local_center_edge_uz_mm": float(depth[center].mean() - depth[edge].mean()),
    }


def load_cases(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    raw = json.load(open(path))
    return {int(row["frame"]): row for row in raw}


def strength_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("strength_") or part.startswith("strength"):
            return part.replace("strength_", "").replace("strength", "")
    return "unknown"


def frame_from_path(path: Path) -> int | None:
    m = re.search(r"frame_(\d+)", str(path))
    return int(m.group(1)) if m else None


def analyze_file(path: Path, cases: dict[int, dict]) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    coords = np.asarray(z["coords"], dtype=np.float64)
    fields = _as_rows(z["disp"])
    frame_idx = frame_from_path(path)
    case = cases.get(frame_idx, {})
    rows = []
    for i, field in enumerate(fields):
        row = {
            "path": str(path),
            "strength_label": strength_from_path(path),
            "frame": int(frame_idx if frame_idx is not None else i),
            "case": case.get("case", f"frame_{frame_idx if frame_idx is not None else i:03d}"),
            "depth_param_mm": float(np.asarray(z["params"])[i, 2] * 1e3),
            "radius_param_mm": float(np.asarray(z["params"])[i, 3] * 1e3),
            "shear_param_mm": float(np.linalg.norm(np.asarray(z["params"])[i, 4:6]) * 1e3),
            "mode": int(np.asarray(z["mode"])[i]),
            "solve_time_s": float(np.asarray(z["solve_time_s"]).reshape(-1)[min(i, len(np.asarray(z["solve_time_s"]).reshape(-1)) - 1)]),
        }
        for key in ("gel_constraint_strength", "indentor_constraint_strength"):
            if key in z.files:
                row[key] = float(np.asarray(z[key]).reshape(-1)[0])
        if "gel_bottom_bc" in z.files:
            row["gel_bottom_bc"] = str(np.asarray(z["gel_bottom_bc"]).reshape(-1)[0])
        row.update(field_metrics(field, coords))
        rows.append(row)
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    def strength_label(value) -> str:
        try:
            return f"{float(value):.0f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Phase 7 BC Sweep Diagnostic",
        "",
        "| Case | Gel BC | Gel strength | Indentor strength | Uniform energy | Mean shift | Residual mean | Residual p95 | Depth max | Depth p95 | Local uz | Solve |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["case"],
            r.get("gel_bottom_bc", "soft"),
            float(r.get("gel_constraint_strength", 0.0)),
            float(r.get("indentor_constraint_strength", 0.0)),
        ),
    )
    for r in rows_sorted:
        gel_strength = r.get("gel_constraint_strength", r["strength_label"])
        indentor_strength = r.get("indentor_constraint_strength")
        lines.append(
            f"| `{r['case']}` "
            f"| `{r.get('gel_bottom_bc', 'soft')}` "
            f"| `{strength_label(gel_strength)}` "
            f"| `{strength_label(indentor_strength) if indentor_strength is not None else 'n/a'}` "
            f"| `{100*r['uniform_energy_fraction']:.2f}%` "
            f"| `{r['mean_shift_mag_mm']:.3f} mm` "
            f"| `{r['residual_tangential_mean_mm']:.3f} mm` "
            f"| `{r['residual_tangential_p95_mm']:.3f} mm` "
            f"| `{r['depth_max_mm']:.3f} mm` "
            f"| `{r['depth_p95_mm']:.3f} mm` "
            f"| `{r['local_center_edge_uz_mm']:.3f} mm` "
            f"| `{r['solve_time_s']:.2f}s` |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="npz files to analyze")
    ap.add_argument("--root", help="root directory; analyzes all uipc_gt_shear.npz below it")
    ap.add_argument("--cases-json", help="optional case metadata JSON")
    ap.add_argument("--out-json", help="write machine-readable summary JSON")
    ap.add_argument("--out-md", help="write markdown summary table")
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]
    if args.root:
        paths.extend(sorted(Path(args.root).glob("**/uipc_gt_shear.npz")))
    if not paths:
        raise SystemExit("no npz paths provided")

    cases = load_cases(Path(args.cases_json) if args.cases_json else None)
    rows: list[dict] = []
    for path in paths:
        rows.extend(analyze_file(path, cases))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, "w"), indent=2, default=float)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, out)

    if not args.out_json and not args.out_md:
        print(json.dumps(rows, indent=2, default=float))
    else:
        print(f"diagnosed {len(rows)} fields")


if __name__ == "__main__":
    main()
