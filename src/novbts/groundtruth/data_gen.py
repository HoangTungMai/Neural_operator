#!/usr/bin/env python3
"""
Phase 3 dataset generation.

Ground-truth physics comes from gt_hertz_mindlin (semi-analytic Hertz +
Cattaneo-Mindlin).  This is the Phase-3 fallback/validator GT; once the
Isaac Sim PhysX-FEM extractor (isaac_extract_groundtruth.py) is available
it writes the SAME schema, so train/eval scripts need no change.

Saves to data/phase3_gt/ in .npz format (schema from the roadmap):
  params  [N, 9]   cx,cy,depth,radius,shear_x,shear_y,mu,stiffness,geom
  coords  [M, 2]   marker grid (normalised -1..1)
  disp    [N,M,3]  displacement field (ux,uy,uz)
  mode    [N]      0=normal 1=stick 2=partial_slip 3=full_slip (physical)

Canonical splits: train, val, test_id, test_slip.
OOD splits (RQ2): radius / depth / material / friction / geometry / resolution
outside the training range.
"""

import argparse
from pathlib import Path

import numpy as np

from novbts.groundtruth.hertz_mindlin import MODE_NAMES, hertz_mindlin_field
from novbts.paths import ANALYTIC

PARAMS_DIM = 9

# Training-distribution ranges (OOD splits sample OUTSIDE these).
RANGES = {
    "cx":        (-0.7, 0.7),
    "cy":        (-0.7, 0.7),
    "depth":     (0.1, 1.0),
    "radius":    (0.08, 0.33),
    "mu":        (0.3, 0.9),
    "stiffness": (0.5, 3.5),
}


def make_grid(side: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)


def _u(rng, lo, hi, n):
    return rng.uniform(lo, hi, size=n)


def sample_params(rng, n_per_mode, overrides=None, modes=(0, 1, 2, 3)):
    """
    Sample mode-balanced parameters.  `overrides` maps a column name to
    (lo, hi) replacing the default range (used for OOD splits).
    Shear magnitude per mode is built as mu * factor so that the physical
    drive ratio g = |shear|/mu lands in the intended regime.
    """
    ov = overrides or {}

    def rng_for(name):
        lo, hi = ov.get(name, RANGES[name])
        return lo, hi

    factor_by_mode = {
        0: (0.0, 0.02),     # normal
        1: (0.08, 0.35),    # stick
        2: (0.48, 0.72),    # partial slip
        3: (0.90, 1.30),    # full slip
    }
    rows = []
    for mode in modes:
        n = n_per_mode
        p = np.zeros((n, PARAMS_DIM), dtype=np.float64)
        p[:, 0] = _u(rng, *rng_for("cx"), n)
        p[:, 1] = _u(rng, *rng_for("cy"), n)
        p[:, 2] = _u(rng, *rng_for("depth"), n)
        p[:, 3] = _u(rng, *rng_for("radius"), n)
        p[:, 6] = _u(rng, *rng_for("mu"), n)
        p[:, 7] = _u(rng, *rng_for("stiffness"), n)
        p[:, 8] = 1.0 if "geom" in ov else 0.0
        theta = _u(rng, 0.0, 2.0 * np.pi, n)
        f_lo, f_hi = factor_by_mode[mode]
        shear_mag = p[:, 6] * _u(rng, f_lo, f_hi, n)
        p[:, 4] = shear_mag * np.cos(theta)
        p[:, 5] = shear_mag * np.sin(theta)
        rows.append(p)
    params = np.concatenate(rows, axis=0)
    rng.shuffle(params)
    return params


def save_split(path, params, coords):
    disp, mode = hertz_mindlin_field(params, coords)
    np.savez_compressed(
        path,
        params=params.astype(np.float32),
        coords=coords.astype(np.float32),
        disp=disp.astype(np.float32),
        mode=mode.astype(np.int32),
        meta=np.array(
            "gt=hertz_mindlin; units=normalised; schema=params/coords/disp/mode",
            dtype="U120",
        ),
    )
    counts = {MODE_NAMES[i]: int((mode == i).sum()) for i in range(4)}
    size_mb = path.stat().st_size / 1024 ** 2
    print(f"  {path.name:32s} N={params.shape[0]:6d} M={coords.shape[0]:5d} "
          f"{counts} {size_mb:.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-mode", type=int, default=4000)
    ap.add_argument("--val-per-mode", type=int, default=500)
    ap.add_argument("--test-per-mode", type=int, default=500)
    ap.add_argument("--marker-side", type=int, default=32)
    ap.add_argument("--output-dir", default=str(ANALYTIC))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    coords = make_grid(args.marker_side)
    print(f"Grid {args.marker_side}x{args.marker_side}={coords.shape[0]} markers | out={out}\n")

    # canonical splits
    save_split(out / "train.npz",   sample_params(rng, args.train_per_mode), coords)
    save_split(out / "val.npz",     sample_params(rng, args.val_per_mode), coords)
    save_split(out / "test_id.npz", sample_params(rng, args.test_per_mode), coords)
    # slip-only test (partial + full)
    save_split(out / "test_slip.npz",
               sample_params(rng, args.test_per_mode, modes=(2, 3)), coords)

    # OOD parameter splits (RQ2) — sampled OUTSIDE training ranges
    ood = [
        ("small_radius",  {"radius": (0.03, 0.07)}),
        ("large_radius",  {"radius": (0.34, 0.50)}),
        ("deep_indent",   {"depth": (1.01, 1.60)}),
        ("soft_material", {"stiffness": (0.05, 0.45)}),
        ("low_friction",  {"mu": (0.05, 0.28)}),
        ("flat_geom",     {"geom": (0, 1)}),
    ]
    for name, overrides in ood:
        save_split(out / f"test_ood_{name}.npz",
                   sample_params(rng, args.test_per_mode, overrides=overrides), coords)

    # resolution OOD (same params, different grid)
    base = sample_params(rng, args.test_per_mode)
    for res in (16, 64):
        save_split(out / f"test_ood_res{res}.npz", base, make_grid(res))

    print(f"\nDone. {len(list(out.glob('*.npz')))} files in {out}")


if __name__ == "__main__":
    main()
