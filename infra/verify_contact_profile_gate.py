#!/usr/bin/env python3
"""Tier-4 non-tautology gate for mesh-raycast contact profiles.

This exercises the same analytic-free raycast path used for mesh objects on a
curved sphere mesh, then compares it with the analytic Hertzian/parabolic cap.
An exactly-zero difference is a failure: it means the gate is comparing an
analytic profile to itself instead of validating the mesh path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "novbts" / "groundtruth"))

from contact_imprint import (  # noqa: E402
    mesh_contact_profile,
    mesh_object_surface,
    sphere_parabolic_penetration,
)


def _mesh_validity(vertices: np.ndarray, faces: np.ndarray) -> tuple[bool, dict[str, float | int]]:
    edges = {}
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges[key] = edges.get(key, 0) + 1
    nonmanifold = sum(1 for c in edges.values() if c != 2)

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    center = np.zeros((1, 3), dtype=np.float64)
    pts = np.vstack([V, center])
    ci = len(V)
    tets = np.array([[int(a), int(b), int(c), ci] for a, b, c in F], dtype=np.int64)
    p = pts[tets]
    vol6 = np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    )
    min_abs_vol6 = float(np.min(np.abs(vol6))) if vol6.size else 0.0
    ok = (nonmanifold == 0) and (min_abs_vol6 > 1.0e-18)
    return ok, {
        "vertices": int(V.shape[0]),
        "faces": int(F.shape[0]),
        "nonmanifold_edges": int(nonmanifold),
        "min_abs_vol6": min_abs_vol6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz",
                    help="sphere IPC npz used to borrow params/coords for the gate")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--object", default="sphere_mesh",
                    help="mesh object name to validate (default preserves the historical sphere gate)")
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--max-rel-l2", type=float, default=0.05)
    ap.add_argument("--depth", type=float, default=-1.0,
                    help="override indentation depth for the borrowed params row")
    ap.add_argument("--radius", type=float, default=-1.0,
                    help="override radius for the borrowed params row")
    ap.add_argument("--half-z", type=float, default=-1.0,
                    help="mesh half-height; default uses radius")
    ap.add_argument("--r2", type=float, default=-1.0,
                    help="secondary radius for anisotropic mesh objects")
    ap.add_argument("--flat-ok", action="store_true",
                    help="allow a flat non-varying positive profile")
    args = ap.parse_args()

    src = np.load(args.data, allow_pickle=True)
    params = np.asarray(src["params"], dtype=np.float32)
    if args.frame < 0 or args.frame >= params.shape[0]:
        raise SystemExit(f"--frame must be in [0, {params.shape[0] - 1}], got {args.frame}")
    row = params[args.frame].copy()
    if args.depth > 0:
        row[2] = args.depth
    if args.radius > 0:
        row[3] = args.radius
    row[8] = 3.0
    half_z = float(args.half_z if args.half_z > 0 else row[3])
    r2 = float(args.r2 if args.r2 > 0 else row[3])
    if row.shape[0] > 9:
        row[9] = r2
    coords = np.asarray(src["coords"], dtype=np.float32)

    V, F, bottom_off = mesh_object_surface(
        args.object,
        float(row[3]),
        half_z,
        r2=r2,
        subdiv=args.subdiv,
    )
    mesh_ok, mesh_stats = _mesh_validity(V, F)
    raycast = mesh_contact_profile(
        coords,
        row,
        args.object,
        half_z=half_z,
        r2=r2,
        subdiv=args.subdiv,
    )
    rel_l2 = float("nan")
    max_abs_diff = float("nan")
    if args.object in ("sphere", "sphere_mesh", "curved_sphere"):
        analytic = sphere_parabolic_penetration(coords, row)
        diff = raycast - analytic
        max_abs_diff = float(np.max(np.abs(diff)))
        rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / (np.linalg.norm(analytic.reshape(-1)) + 1e-12))
    positive = raycast[raycast > 0.0]
    max_profile = float(np.max(raycast)) if raycast.size else 0.0
    std_pos = float(np.std(positive)) if positive.size else 0.0
    positive_frac = float(positive.size / raycast.size) if raycast.size else 0.0
    print(
        "CONTACT_PROFILE_GATE",
        f"object={args.object}",
        f"source={args.data}",
        f"frame={args.frame}",
        f"subdiv={args.subdiv}",
        f"depth={float(row[2]):.9e}",
        f"R={float(row[3]):.9e}",
        f"half_z={half_z:.9e}",
        f"bottom_off={float(bottom_off):.9e}",
        f"vertices={mesh_stats['vertices']}",
        f"faces={mesh_stats['faces']}",
        f"nonmanifold_edges={mesh_stats['nonmanifold_edges']}",
        f"min_abs_vol6={mesh_stats['min_abs_vol6']:.9e}",
        f"max_profile={max_profile:.9e}",
        f"std_pos={std_pos:.9e}",
        f"positive_frac={positive_frac:.9e}",
        f"max_abs_diff={max_abs_diff:.9e}",
        f"rel_l2={rel_l2:.9e}",
        flush=True,
    )
    if args.object in ("sphere", "sphere_mesh", "curved_sphere"):
        profile_ok = (0.0 < max_abs_diff) and (rel_l2 < args.max_rel_l2)
    elif args.flat_ok:
        profile_ok = max_profile > 0.0
    else:
        profile_ok = (max_profile > 0.0) and (std_pos > 0.0)
    ok = mesh_ok and profile_ok
    print("CONTACT_PROFILE_GATE_OK" if ok else "CONTACT_PROFILE_GATE_FAIL", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
