#!/usr/bin/env python3
"""Average repeated UIPC runs into PhysX-compatible GT frames.

UIPC/libuipc is not bitwise deterministic on GPU, so Phase 7 production GT uses
K independent runs per contact frame and averages the displacement field. This
script supports both:

  * one frame: average files matched by --glob into --out;
  * a sweep: average every combo_*/frame_*/rep_*/uipc_gt_shear.npz under
    --sweep-dir, write per-frame averages, and concatenate them into --out.

The output schema stays compatible with the FNO/FEM pipeline:
params, coords, disp, mode, solve_time_s, meta, plus IPC provenance fields.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np


G_STICK, G_PARTIAL, G_FULL = 0.04, 0.48, 1.0


def label_mode_from_g(g: float) -> int:
    if g < G_STICK:
        return 0
    if g < G_PARTIAL:
        return 1
    if g < G_FULL:
        return 2
    return 3


def infer_mode(params: np.ndarray, shear_scale: float) -> np.ndarray:
    """Infer PhysX-style mode from the sweep convention |shear| = g * mu * scale."""
    p = np.asarray(params)
    shear = np.linalg.norm(p[:, 4:6], axis=1)
    mu = np.clip(p[:, 6], 1e-9, None)
    g = shear / (mu * shear_scale)
    return np.array([label_mode_from_g(float(x)) for x in g], dtype=np.int32)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def replicate_noise(fields: np.ndarray) -> dict[str, float]:
    """Mean pairwise rel-L2 across replicate fields, split by channel."""
    if fields.shape[0] < 2:
        return {"rep_noise_overall": 0.0, "rep_noise_normal": 0.0, "rep_noise_tangential": 0.0}
    vals_o, vals_n, vals_t = [], [], []
    for i in range(fields.shape[0]):
        for j in range(i + 1, fields.shape[0]):
            vals_o.append(rel_l2(fields[i], fields[j]))
            vals_n.append(rel_l2(fields[i, :, 2], fields[j, :, 2]))
            vals_t.append(rel_l2(fields[i, :, :2], fields[j, :, :2]))
    return {
        "rep_noise_overall": float(np.mean(vals_o)),
        "rep_noise_normal": float(np.mean(vals_n)),
        "rep_noise_tangential": float(np.mean(vals_t)),
    }


def _same(a: np.ndarray, b: np.ndarray, name: str, path: str) -> None:
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-5, atol=1e-8):
        raise SystemExit(f"{name} mismatch in {path}")


def average_files(paths: list[str], *, mode_shear_scale: float | None = None) -> dict[str, np.ndarray]:
    if not paths:
        raise SystemExit("no replicate files matched")
    loaded = [np.load(p, allow_pickle=True) for p in paths]
    ref = loaded[0]
    for p, d in zip(paths[1:], loaded[1:]):
        _same(ref["params"], d["params"], "params", p)
        _same(ref["coords"], d["coords"], "coords", p)
        for key in ("gel_res", "eps_velocity", "d_hat", "contact_resistance"):
            if key in ref.files and key in d.files:
                _same(np.asarray(ref[key]), np.asarray(d[key]), key, p)

    fields = np.stack([np.asarray(d["disp"], dtype=np.float32)[0] for d in loaded], axis=0)
    mean_field = fields.mean(axis=0, dtype=np.float64).astype(np.float32)
    noise = replicate_noise(fields)

    params = np.asarray(ref["params"], dtype=np.float32)
    modes = np.asarray(ref["mode"], dtype=np.int32)
    if mode_shear_scale is not None and (modes < 0).any():
        modes = infer_mode(params, mode_shear_scale)

    solve_rep = np.concatenate([np.asarray(d["solve_time_s"], dtype=np.float32).reshape(-1) for d in loaded])
    out: dict[str, np.ndarray] = {
        "params": params,
        "coords": np.asarray(ref["coords"], dtype=np.float32),
        "disp": mean_field[None, ...].astype(np.float32),
        "mode": modes.astype(np.int32),
        # Production cost for an averaged frame is K solver calls.
        "solve_time_s": np.array([solve_rep.sum()], dtype=np.float32),
        "rep_solve_time_s": solve_rep.astype(np.float32),
        "rep_solve_time_mean_s": np.array([solve_rep.mean()], dtype=np.float32),
        "rep_solve_time_sum_s": np.array([solve_rep.sum()], dtype=np.float32),
        "n_replicates": np.array([len(paths)], dtype=np.int32),
        "source_files": np.array(paths, dtype="U512"),
        "meta": np.array(
            "gt=uipc_ipc_SHEAR; rep_averaged; tacex_uipc; units=m; schema=params/coords/disp/mode",
            dtype="U120",
        ),
    }
    for key in ("gel_res", "eps_velocity", "d_hat", "contact_resistance", "n_tet_verts"):
        if key in ref.files:
            out[key] = np.asarray(ref[key]).copy()
    for key, value in noise.items():
        out[key] = np.array([value], dtype=np.float32)
    return out


def save_npz(path: str | Path, data: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def aggregate_sweep(sweep_dir: str, out_path: str, *, mode_shear_scale: float | None,
                    expect_reps: int | None = None,
                    frame_out_name: str = "uipc_gt_shear_avg.npz") -> None:
    frame_dirs = sorted(Path(sweep_dir).glob("combo_*/frame_*"))
    rows, skipped = [], []
    for frame_dir in frame_dirs:
        rep_paths = sorted(str(p) for p in frame_dir.glob("rep_*/uipc_gt_shear.npz"))
        if not rep_paths:
            skipped.append(str(frame_dir))
            continue
        if expect_reps is not None and len(rep_paths) != expect_reps:
            skipped.append(f"{frame_dir} ({len(rep_paths)}/{expect_reps} reps)")
            continue
        avg = average_files(rep_paths, mode_shear_scale=mode_shear_scale)
        frame_out = frame_dir / frame_out_name
        save_npz(frame_out, avg)
        rows.append(avg)

    if not rows:
        raise SystemExit(f"no complete frame dirs found under {sweep_dir}")

    coords = rows[0]["coords"]
    for i, row in enumerate(rows[1:], start=1):
        _same(coords, row["coords"], "coords", f"frame {i}")

    merged: dict[str, np.ndarray] = {
        "params": np.concatenate([r["params"] for r in rows], axis=0).astype(np.float32),
        "coords": coords.astype(np.float32),
        "disp": np.concatenate([r["disp"] for r in rows], axis=0).astype(np.float32),
        "mode": np.concatenate([r["mode"] for r in rows], axis=0).astype(np.int32),
        "solve_time_s": np.concatenate([r["solve_time_s"] for r in rows], axis=0).astype(np.float32),
        "n_replicates": np.concatenate([r["n_replicates"] for r in rows], axis=0).astype(np.int32),
        "rep_noise_overall": np.concatenate([r["rep_noise_overall"] for r in rows], axis=0).astype(np.float32),
        "rep_noise_normal": np.concatenate([r["rep_noise_normal"] for r in rows], axis=0).astype(np.float32),
        "rep_noise_tangential": np.concatenate([r["rep_noise_tangential"] for r in rows], axis=0).astype(np.float32),
        "meta": np.array("gt=uipc_ipc_SHEAR_sweep; rep_averaged; tacex_uipc; units=m", dtype="U96"),
    }
    for key in ("gel_res", "eps_velocity", "d_hat", "contact_resistance", "n_tet_verts"):
        if key in rows[0]:
            merged[key] = np.concatenate([r[key] for r in rows], axis=0)
    save_npz(out_path, merged)
    print(f"aggregated {len(rows)} averaged UIPC frames -> {out_path}")
    if skipped:
        print(f"skipped {len(skipped)} frame dirs with no reps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", dest="glob_pattern", default=None,
                    help="replicate file glob for one averaged frame")
    ap.add_argument("--sweep-dir", default=None,
                    help="directory with combo_*/frame_*/rep_*/uipc_gt_shear.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode-shear-scale", type=float, default=None,
                    help="infer mode from |shear|/(mu*scale), e.g. PhysX convention 0.01")
    ap.add_argument("--expect-reps", type=int, default=None,
                    help="when aggregating --sweep-dir, skip frame dirs that do not have this many reps")
    args = ap.parse_args()

    if bool(args.glob_pattern) == bool(args.sweep_dir):
        raise SystemExit("pass exactly one of --glob or --sweep-dir")

    if args.glob_pattern:
        paths = sorted(glob.glob(args.glob_pattern))
        data = average_files(paths, mode_shear_scale=args.mode_shear_scale)
        save_npz(args.out, data)
        print(f"averaged {len(paths)} UIPC reps -> {args.out}")
        print(f"mode={data['mode'].tolist()} rep_noise_tang={float(data['rep_noise_tangential'][0]):.4f}")
    else:
        aggregate_sweep(args.sweep_dir, args.out, mode_shear_scale=args.mode_shear_scale,
                        expect_reps=args.expect_reps)


if __name__ == "__main__":
    main()
