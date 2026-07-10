"""Pure gel-frame / placement geometry for the grasp demo.

All functions here are parameter-pure (no sim state, no args capture) so they
are host-testable. Frames are dicts with "origin"/"x"/"y"/"normal" world-space
lists, as produced by the driver's current_gel_frames_w().
"""
from __future__ import annotations

import numpy as np


def gel_gap_from_frames(frames):
    if "left" not in frames or "right" not in frames:
        return {"frames": frames, "midpoint": None, "gap_m": None}
    left = np.asarray(frames["left"]["origin"], dtype=np.float64)
    right = np.asarray(frames["right"]["origin"], dtype=np.float64)
    left_n = np.asarray(frames["left"]["normal"], dtype=np.float64)
    right_n = np.asarray(frames["right"]["normal"], dtype=np.float64)
    left_n = left_n / (np.linalg.norm(left_n) + 1.0e-12)
    right_n = right_n / (np.linalg.norm(right_n) + 1.0e-12)
    origin_delta = right - left
    left_face_gap = float(origin_delta @ left_n)
    right_face_gap = float((-origin_delta) @ right_n)
    face_gap = 0.5 * (left_face_gap + right_face_gap)
    origin_gap = float(np.linalg.norm(left - right))
    normal_dot = float(left_n @ right_n)
    return {
        "frames": frames,
        "midpoint": ((left + right) * 0.5).tolist(),
        "gap_m": float(face_gap) if np.isfinite(face_gap) else None,
        "origin_gap_m": origin_gap if np.isfinite(origin_gap) else None,
        "left_face_gap_m": left_face_gap if np.isfinite(left_face_gap) else None,
        "right_face_gap_m": right_face_gap if np.isfinite(right_face_gap) else None,
        "normal_dot": normal_dot if np.isfinite(normal_dot) else None,
    }


def T_from_gel_frame(frame):
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = np.asarray(frame["x"], dtype=np.float64)
    T[:3, 1] = np.asarray(frame["y"], dtype=np.float64)
    T[:3, 2] = np.asarray(frame["normal"], dtype=np.float64)
    T[:3, 3] = np.asarray(frame["origin"], dtype=np.float64)
    return T


def object_center_for_opposing_normals(origins, normals, supports=None):
    if len(origins) != 2 or len(normals) != 2:
        return None
    n0 = np.asarray(normals[0], dtype=np.float64)
    n1 = np.asarray(normals[1], dtype=np.float64)
    n0 = n0 / (np.linalg.norm(n0) + 1.0e-12)
    n1 = n1 / (np.linalg.norm(n1) + 1.0e-12)
    if float(n0 @ n1) > -0.95:
        return None
    o0 = np.asarray(origins[0], dtype=np.float64)
    o1 = np.asarray(origins[1], dtype=np.float64)
    p = 0.5 * (o0 + o1)
    # For opposing gel normals, place the object at the live face
    # midpoint. Per-pad support extents can differ when the two gel
    # frames are not perfectly mirrored; adding that difference here
    # creates a deterministic off-center bias along the grip axis.
    coord = 0.5 * (float(n0 @ o0) + float(n0 @ o1))
    return p + n0 * (coord - float(n0 @ p))


def pen_stat_values(pen_stats):
    return {
        side: float(stats.get("stat_m", 0.0))
        for side, stats in (pen_stats or {}).items()
    }


def pen_max_values(pen_stats):
    return {
        side: float(stats.get("max_m", 0.0))
        for side, stats in (pen_stats or {}).items()
    }


def placement_pen_summary(pen, target_depth):
    vals = [float(pen[side]) for side in ("left", "right") if side in pen]
    if not vals:
        return {
            "min_pen_m": None,
            "max_pen_m": None,
            "mean_pen_m": None,
            "pen_error_m": None,
            "mean_pen_delta_m": None,
            "imbalance_m": None,
        }
    target = float(target_depth)
    imbalance = abs(float(pen.get("left", vals[0])) - float(pen.get("right", vals[-1]))) if len(vals) >= 2 else 0.0
    mean_pen = float(sum(vals) / len(vals))
    return {
        "min_pen_m": float(min(vals)),
        "max_pen_m": float(max(vals)),
        "mean_pen_m": mean_pen,
        "pen_error_m": float(max(abs(v - target) for v in vals)),
        "mean_pen_delta_m": float(mean_pen - target),
        "imbalance_m": float(imbalance),
    }


def placement_balance_tolerance(target_depth):
    return max(5.0e-5, 0.25 * float(target_depth))


def grip_axis_for_gel_frames(frames):
    if not frames or "left" not in frames or "right" not in frames:
        return None
    n_l = np.asarray(frames["left"]["normal"], dtype=np.float64)
    n_r = np.asarray(frames["right"]["normal"], dtype=np.float64)
    n_l = n_l / (np.linalg.norm(n_l) + 1.0e-12)
    n_r = n_r / (np.linalg.norm(n_r) + 1.0e-12)
    axis = n_l - n_r
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return None
    return axis / norm
