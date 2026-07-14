"""Analytic penetration-field helpers shared by the grasp demo tactile path."""
from __future__ import annotations

import numpy as np

from novbts.simulation.runtime.gel_contact import pen_cuboid, pen_cylinder, pen_sphere, pose_to_T


def pen_for_object(kind: str, size: float, T_gel_obj: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(T_gel_obj)):
        side = int(round(np.sqrt(grid.shape[0])))
        return np.zeros((side, side), dtype=np.float32)
    if kind == "sphere":
        return pen_sphere(T_gel_obj, size / 2.0, grid)
    if kind == "cube":
        return pen_cuboid(T_gel_obj, (size, size, size), grid)
    if kind == "cylinder":
        return pen_cylinder(T_gel_obj, (size / 2.0, size), grid)
    raise ValueError(kind)


def pen_grid_stats(pen: np.ndarray) -> dict:
    arr = np.asarray(pen, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "stat_m": 0.0,
            "max_m": 0.0,
            "p90_m": 0.0,
            "mean_contact_m": 0.0,
            "contact_fraction": 0.0,
        }
    finite = np.maximum(finite, 0.0)
    contact = finite[finite > 0.0]
    if contact.size == 0:
        stat = 0.0
        p90 = 0.0
        mean_contact = 0.0
    else:
        mean_contact = float(contact.mean())
        p90 = float(np.percentile(contact, 90.0))
        # Guard metric: mean positive contact depth. The raw max is still
        # recorded separately for detecting pathological analytic overlap.
        stat = mean_contact
    return {
        "stat_m": float(stat),
        "max_m": float(finite.max()),
        "p90_m": float(p90),
        "mean_contact_m": float(mean_contact),
        "contact_fraction": float(contact.size / finite.size),
    }


def support_extent_for_object(kind: str, size: float, normal_w, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> float:
    """Half extent of the rigid object along a world-space contact normal."""
    n_w = np.asarray(normal_w, dtype=np.float64)
    n_w = n_w / (np.linalg.norm(n_w) + 1.0e-12)
    R_w_obj = pose_to_T((0.0, 0.0, 0.0), quat_wxyz)[:3, :3]
    n_obj = R_w_obj.T @ n_w
    half = float(size) / 2.0
    if kind == "sphere":
        return half
    if kind == "cube":
        return half * float(np.abs(n_obj).sum())
    if kind == "cylinder":
        radial = half * float(np.linalg.norm(n_obj[:2]))
        axial = half * float(abs(n_obj[2]))
        return radial + axial
    raise ValueError(kind)
