"""Pure-numpy rotation/quaternion helpers and CLI vector parsers for the grasp demo.

Quaternions are wxyz throughout (Isaac/USD convention). No Isaac imports here —
host-testable.
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_from_R(R: np.ndarray) -> list[float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def quat_mul_wxyz(a, b) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    q = np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1.0e-12)


def quat_conj_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_angle_error_wxyz(target, actual) -> float:
    q_rel = quat_mul_wxyz(target, quat_conj_wxyz(actual))
    w = abs(float(np.clip(q_rel[0], -1.0, 1.0)))
    return float(2.0 * np.arccos(w))


def quat_between_vectors_wxyz(src, dst) -> np.ndarray:
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1.0e-12)
    b = b / (np.linalg.norm(b) + 1.0e-12)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 1.0 - 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1.0e-9:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1.0e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / (np.linalg.norm(axis) + 1.0e-12)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    axis = np.cross(a, b)
    q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1.0e-12)


def Rx(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def Ry(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def Rz(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_from_rpy_deg(rpy: str) -> list[float]:
    vals = [float(x.strip()) for x in rpy.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--robot-root-rpy-deg must be 'roll,pitch,yaw'")
    roll, pitch, yaw = [np.deg2rad(v) for v in vals]
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return quat_wxyz_from_R(R)


def vec3_arg(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("expected comma-separated x,y,z")
    return np.asarray(vals, dtype=np.float64)


def float_list_arg(text: str, expected: int) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    return vals


def gel_R_from_normal(normal_w: np.ndarray) -> np.ndarray:
    z = np.asarray(normal_w, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1.0e-12)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1.0e-12)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])
