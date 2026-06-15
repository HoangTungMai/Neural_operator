#!/usr/bin/env python3
"""Aggregate per-combo FEM sweep npz files into one swept dataset.

Each combo dir (data/fem/sweep*/combo_NNN/fem_gt_shear.npz) holds 40 frames with
a shared marker grid (coords). This concatenates params/disp/mode/solve_time
across all combos that hold the full frame count, taking coords from the first.

Usage:
  python -m novbts.groundtruth.aggregate_sweep \
      --sweep-dir data/fem/sweep32 --out data/fem/shear_fine_swept_res32.npz
"""
import argparse
import glob
import os

import numpy as np


def aggregate(sweep_dir, expect_frames=40):
    combos = sorted(glob.glob(os.path.join(sweep_dir, "combo_*", "fem_gt_shear.npz")))
    params, disps, modes, solve = [], [], [], []
    coords = None
    used, skipped = [], []
    for path in combos:
        d = np.load(path, allow_pickle=True)
        n = d["params"].shape[0]
        if n != expect_frames:
            skipped.append((path, n))
            continue
        if coords is None:
            coords = d["coords"]
        params.append(d["params"])
        disps.append(d["disp"])
        modes.append(d["mode"])
        if "solve_time_s" in d.files:
            solve.append(d["solve_time_s"])
        used.append(path)
    if not used:
        raise SystemExit(f"no complete combos ({expect_frames} frames) in {sweep_dir}")
    out = {
        "params": np.concatenate(params, 0).astype(np.float32),
        "coords": coords.astype(np.float32),
        "disp": np.concatenate(disps, 0).astype(np.float32),
        "mode": np.concatenate(modes, 0).astype(np.int32),
    }
    if solve:
        out["solve_time_s"] = np.concatenate(solve, 0).astype(np.float32)
    return out, used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-frames", type=int, default=40)
    ap.add_argument("--merge-base", default=None,
                    help="existing swept npz to concatenate with this sweep "
                         "(e.g. add pure-normal frames to shear_fine_swept.npz)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle the merged dataset with this seed so the added "
                         "frames distribute across the train/test split")
    args = ap.parse_args()
    out, used, skipped = aggregate(args.sweep_dir, args.expect_frames)

    if args.merge_base:
        base = np.load(args.merge_base, allow_pickle=True)
        if base["coords"].shape != out["coords"].shape:
            raise SystemExit(f"coords mismatch: base {base['coords'].shape} vs "
                             f"sweep {out['coords'].shape} — different marker grids")
        nb, nn = base["params"].shape[0], out["params"].shape[0]
        merged = {
            "params": np.concatenate([base["params"], out["params"]], 0).astype(np.float32),
            "coords": out["coords"].astype(np.float32),
            "disp": np.concatenate([base["disp"], out["disp"]], 0).astype(np.float32),
            "mode": np.concatenate([base["mode"], out["mode"]], 0).astype(np.int32),
        }
        print(f"merged base {nb} + sweep {nn} = {merged['params'].shape[0]} frames")
        out = merged

    if args.shuffle_seed is not None:
        perm = np.random.default_rng(args.shuffle_seed).permutation(out["params"].shape[0])
        for k in ("params", "disp", "mode"):
            out[k] = out[k][perm]
        if "solve_time_s" in out:
            out["solve_time_s"] = out["solve_time_s"][perm]
        print(f"shuffled {len(perm)} frames with seed {args.shuffle_seed}")

    np.savez_compressed(args.out, **out)
    print(f"aggregated {len(used)} combos -> {out['params'].shape[0]} frames -> {args.out}")
    print(f"coords side={int(round(out['coords'].shape[0] ** 0.5))}")
    import collections
    print(f"mode dist: {dict(collections.Counter(out['mode'].tolist()))}")
    if "solve_time_s" in out:
        st = out["solve_time_s"]
        print(f"solve_time_s mean={st.mean():.2f}s  ({1.0/st.mean():.3f} fps)")
    for path, n in skipped:
        print(f"  skipped incomplete: {path} ({n} frames)")


if __name__ == "__main__":
    main()
