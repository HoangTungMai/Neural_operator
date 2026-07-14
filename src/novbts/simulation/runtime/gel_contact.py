#!/usr/bin/env python3
"""Analytic rigid-object penetration maps in the VBTS gel frame.

All functions are numpy-only by design. Isaac drivers can feed object poses into
these helpers, while host tests can exercise the contact math without importing
Isaac.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from novbts.research.fno.field2field import params_to_fieldinput


def gel_grid(side: int = 32, half: float = 0.009) -> np.ndarray:
    """Return row-major ``[side*side,2]`` xy points on ``[-half, half]`` metres."""
    xs = np.linspace(-half, half, side, dtype=np.float64)
    ys = np.linspace(-half, half, side, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def _quat_wxyz_to_R(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must be wxyz with 4 values, got {q}")
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _grid_xyz(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    return np.column_stack([g[:, 0], g[:, 1], np.zeros(g.shape[0], dtype=np.float64)])


def _reshape_pen(pen: np.ndarray, grid: np.ndarray) -> np.ndarray:
    side = int(round(np.sqrt(np.asarray(grid).shape[0])))
    if side * side != np.asarray(grid).shape[0]:
        raise ValueError("grid must be square and row-major")
    return pen.reshape(side, side).astype(np.float32)


def _ray_setup(T_gel_obj: np.ndarray, grid: np.ndarray):
    R = np.asarray(T_gel_obj, dtype=np.float64)[:3, :3]
    t = np.asarray(T_gel_obj, dtype=np.float64)[:3, 3]
    origins_g = _grid_xyz(grid)
    origins_o = (origins_g - t[None]) @ R
    direction_o = R.T @ np.array([0.0, 0.0, 1.0])
    return origins_o, direction_o


def _ray_sphere_t_enter(T_gel_obj: np.ndarray, radius: float, grid: np.ndarray) -> np.ndarray:
    c = np.asarray(T_gel_obj, dtype=np.float64)[:3, 3]
    x = grid[:, 0] - c[0]
    y = grid[:, 1] - c[1]
    rr = np.asarray(radius, dtype=np.float64) ** 2 - x * x - y * y
    root = np.sqrt(np.maximum(rr, 0.0))
    t_enter = c[2] - root
    return np.where(rr >= 0.0, t_enter, np.inf)


def pen_sphere(
    T_gel_obj: np.ndarray,
    radius: float,
    grid: np.ndarray | None = None,
    *,
    clip: float = 0.00075,
    exact: bool = False,
) -> np.ndarray:
    grid = gel_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if exact:
        t_enter = _ray_sphere_t_enter(T_gel_obj, radius, grid)
        pen = np.where(t_enter < 0.0, -t_enter, 0.0)
    else:
        # The LR-FNO was trained with field2field.params_to_fieldinput's
        # paraboloid sphere profile. Use that convention by default so Isaac
        # contact maps land on the same input distribution as the checkpoint.
        c = np.asarray(T_gel_obj, dtype=np.float64)[:3, 3]
        depth = float(radius) - float(c[2])
        rho2 = (grid[:, 0] - c[0]) ** 2 + (grid[:, 1] - c[1]) ** 2
        pen = np.maximum(0.0, depth - rho2 / (2.0 * float(radius)))
    return _reshape_pen(np.clip(pen, 0.0, clip), grid)


def pen_cuboid(
    T_gel_obj: np.ndarray,
    dims,
    grid: np.ndarray | None = None,
    *,
    clip: float = 0.00075,
) -> np.ndarray:
    """Ray-box penetration for a cuboid with local dimensions ``(x,y,z)``."""
    grid = gel_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    o, d = _ray_setup(T_gel_obj, grid)
    half = np.asarray(dims, dtype=np.float64) / 2.0
    t0 = np.full(o.shape[0], -np.inf, dtype=np.float64)
    t1 = np.full(o.shape[0], np.inf, dtype=np.float64)
    valid = np.ones(o.shape[0], dtype=bool)
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            valid &= np.abs(o[:, ax]) <= half[ax]
            continue
        a = (-half[ax] - o[:, ax]) / d[ax]
        b = (half[ax] - o[:, ax]) / d[ax]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        t0 = np.maximum(t0, lo)
        t1 = np.minimum(t1, hi)
    valid &= t1 >= t0
    valid &= t1 >= 0.0
    pen = np.where(valid & (t0 < 0.0), -t0, 0.0)
    return _reshape_pen(np.clip(pen, 0.0, clip), grid)


def pen_cylinder(
    T_gel_obj: np.ndarray,
    dims,
    grid: np.ndarray | None = None,
    *,
    clip: float = 0.00075,
) -> np.ndarray:
    """Ray-cylinder penetration for a local z-axis cylinder.

    ``dims`` may be ``(radius, height)`` or a dict with ``radius`` and ``height``.
    """
    grid = gel_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if isinstance(dims, dict):
        radius, height = float(dims["radius"]), float(dims["height"])
    else:
        radius, height = float(dims[0]), float(dims[1])
    o, d = _ray_setup(T_gel_obj, grid)
    intervals_lo = np.full(o.shape[0], -np.inf, dtype=np.float64)
    intervals_hi = np.full(o.shape[0], np.inf, dtype=np.float64)

    # Infinite cylinder side interval.
    A = d[0] * d[0] + d[1] * d[1]
    B = 2.0 * (o[:, 0] * d[0] + o[:, 1] * d[1])
    C = o[:, 0] * o[:, 0] + o[:, 1] * o[:, 1] - radius * radius
    valid = np.ones(o.shape[0], dtype=bool)
    if A < 1e-12:
        valid &= C <= 0.0
    else:
        disc = B * B - 4.0 * A * C
        valid &= disc >= 0.0
        root = np.sqrt(np.maximum(disc, 0.0))
        lo = (-B - root) / (2.0 * A)
        hi = (-B + root) / (2.0 * A)
        intervals_lo = np.maximum(intervals_lo, lo)
        intervals_hi = np.minimum(intervals_hi, hi)

    # Finite z slab.
    hz = height / 2.0
    if abs(d[2]) < 1e-12:
        valid &= np.abs(o[:, 2]) <= hz
    else:
        za = (-hz - o[:, 2]) / d[2]
        zb = (hz - o[:, 2]) / d[2]
        intervals_lo = np.maximum(intervals_lo, np.minimum(za, zb))
        intervals_hi = np.minimum(intervals_hi, np.maximum(za, zb))
    valid &= intervals_hi >= intervals_lo
    valid &= intervals_hi >= 0.0
    pen = np.where(valid & (intervals_lo < 0.0), -intervals_lo, 0.0)
    return _reshape_pen(np.clip(pen, 0.0, clip), grid)


def pen_mesh(T_gel_obj: np.ndarray, mesh_path: str, grid: np.ndarray | None = None, *, clip: float = 0.00075):
    """Raycast a mesh with trimesh. This is intentionally lazy-imported."""
    import trimesh

    grid = gel_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    mesh = trimesh.load(mesh_path, force="mesh")
    o, d = _ray_setup(T_gel_obj, grid)
    dirs = np.repeat(d[None], o.shape[0], axis=0)
    loc, idx_ray, _idx_tri = mesh.ray.intersects_location(o, dirs, multiple_hits=True)
    pen = np.zeros(o.shape[0], dtype=np.float64)
    if len(loc):
        t = np.einsum("ij,j->i", loc - o[idx_ray], d)
        for i in np.unique(idx_ray):
            hits = np.sort(t[idx_ray == i])
            below = hits[hits < 0.0]
            above = hits[hits >= 0.0]
            if below.size and above.size:
                pen[i] = -below.min()
    return _reshape_pen(np.clip(pen, 0.0, clip), grid)


@dataclass
class ShearTracker:
    """Track tangential object slip after first contact in gel coordinates."""

    mu: float = 0.6
    reset_after: int = 5
    cap_scale: float = 1.3e-3

    def __post_init__(self) -> None:
        self.anchor_xy: np.ndarray | None = None
        self.miss_count = 0

    def reset(self) -> None:
        self.anchor_xy = None
        self.miss_count = 0

    def update(self, T_gel_obj: np.ndarray, pen: np.ndarray, *, contact_eps: float = 1e-9) -> np.ndarray:
        xy = np.asarray(T_gel_obj, dtype=np.float64)[:2, 3]
        touching = float(np.max(pen)) > contact_eps
        if not touching:
            self.miss_count += 1
            if self.miss_count >= self.reset_after:
                self.reset()
            return np.zeros(2, dtype=np.float32)

        self.miss_count = 0
        if self.anchor_xy is None:
            self.anchor_xy = xy.copy()
        shear = xy - self.anchor_xy
        cap = max(float(self.mu), 0.0) * float(self.cap_scale)
        mag = float(np.linalg.norm(shear))
        if cap > 0.0 and mag > cap:
            shear = shear * (cap / (mag + 1e-12))
        return shear.astype(np.float32)


def _sphere_T_from_depth(radius: float, depth: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return pose_to_T((x, y, radius - depth), (1.0, 0.0, 0.0, 0.0))


def _smoke() -> dict:
    side = 32
    coords = gel_grid(side, 0.009).astype(np.float32)
    max_rel = 0.0
    rows = []
    for R in np.linspace(0.002, 0.006, 5):
        for depth in np.linspace(0.00015, 0.00075, 5):
            T = _sphere_T_from_depth(float(R), float(depth))
            pen = pen_sphere(T, R, coords, clip=0.010)
            params = np.array([[0.0, 0.0, depth, R, 0.0, 0.0, 0.6, 1.0e5, 0.0]], np.float32)
            ref, _ = params_to_fieldinput(params, coords, side)
            rel = float(np.linalg.norm(pen - ref[0, 0]) / (np.linalg.norm(ref[0, 0]) + 1e-12))
            rows.append(rel)
            max_rel = max(max_rel, rel)
    if max_rel >= 0.05:
        raise SystemExit(f"sphere exact-vs-paraboloid rel error too high: {max_rel:.3f}")

    off = pen_sphere(_sphere_T_from_depth(0.004, 0.0005, x=0.002, y=-0.001), 0.004, coords)
    iy, ix = np.unravel_index(int(off.argmax()), off.shape)
    peak_xy = coords.reshape(side, side, 2)[iy, ix]
    if abs(float(peak_xy[0]) - 0.002) > 0.001 or abs(float(peak_xy[1]) + 0.001) > 0.001:
        raise SystemExit(f"offset sphere peak at {peak_xy}, expected near (0.002,-0.001)")

    tr = ShearTracker(mu=0.6)
    zero = np.zeros((side, side), dtype=np.float32)
    p = pen_sphere(_sphere_T_from_depth(0.004, 0.0005), 0.004, coords)
    s0 = tr.update(_sphere_T_from_depth(0.004, 0.0005), p)
    s1 = tr.update(_sphere_T_from_depth(0.004, 0.0005, x=0.0004, y=-0.0002), p)
    for _ in range(5):
        tr.update(_sphere_T_from_depth(0.004, -0.001), zero)
    s2 = tr.update(_sphere_T_from_depth(0.004, 0.0005, x=0.001), p)
    if np.linalg.norm(s0) > 1e-9 or not np.allclose(s1, [0.0004, -0.0002], atol=1e-5):
        raise SystemExit(f"ShearTracker bad slide: s0={s0} s1={s1}")
    if np.linalg.norm(s2) > 1e-9:
        raise SystemExit(f"ShearTracker did not reset after release: {s2}")

    return {"max_rel_sphere_vs_paraboloid": max_rel, "mean_rel": float(np.mean(rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(_smoke())
        print("GEL_CONTACT_SMOKE_OK")


if __name__ == "__main__":
    main()
