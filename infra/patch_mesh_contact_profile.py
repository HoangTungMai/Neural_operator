#!/usr/bin/env python3
"""Offline patch for Tier-4 mesh ``contact_profile`` fields.

The IPC target displacement is left untouched.  Only the stored geometric input
profile is recomputed from the deterministic mesh raycast path.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "novbts" / "groundtruth"))

from contact_imprint import mesh_contact_profile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uipc/geom_ood/mesh/bolt_hex/bolt_hex_avg.npz")
    ap.add_argument("--out", default="", help="output npz; default overwrites --data after making a backup")
    ap.add_argument("--object", default="bolt_hex")
    ap.add_argument("--half-z", type=float, default=-1.0,
                    help="mesh half height; default uses params[:,3] per frame")
    ap.add_argument("--r2-scale", type=float, default=1.0,
                    help="used only when params lack R2 or object wants anisotropy")
    ap.add_argument("--subdiv", type=int, default=2)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out) if args.out else data_path
    d = np.load(data_path, allow_pickle=True)
    payload = {k: np.asarray(d[k]).copy() for k in d.files}
    params = np.asarray(payload["params"], dtype=np.float32)
    coords = np.asarray(payload["coords"], dtype=np.float32)
    old = np.asarray(payload["contact_profile"], dtype=np.float32) if "contact_profile" in payload else None

    profiles = []
    for row in params:
        half_z = float(row[3] if args.half_z <= 0 else args.half_z)
        r2 = float(row[9] if row.shape[0] > 9 else row[3] * args.r2_scale)
        profiles.append(mesh_contact_profile(
            coords,
            row,
            args.object,
            half_z=half_z,
            r2=r2,
            subdiv=args.subdiv,
        ))
    new = np.stack(profiles, axis=0).astype(np.float32)

    rel_l2 = None
    max_abs = None
    if old is not None:
        old = old.reshape(new.shape)
        diff = new - old
        rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / (np.linalg.norm(old.reshape(-1)) + 1e-12))
        max_abs = float(np.max(np.abs(diff)))

    payload["contact_profile"] = new
    payload["contact_profile_name"] = np.array([args.object] * params.shape[0], dtype="U64")

    if out_path == data_path:
        backup = data_path.with_suffix(".PRE_RAYCAST.npz")
        if not backup.exists():
            shutil.copy2(data_path, backup)
        tmp = data_path.with_suffix(".tmp_raycast.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(data_path)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **payload)

    fields = [
        "MESH_CONTACT_PROFILE_PATCHED",
        f"object={args.object}",
        f"data={data_path}",
        f"out={out_path}",
        f"frames={params.shape[0]}",
    ]
    if rel_l2 is not None:
        fields += [f"old_vs_raycast_rel_l2={rel_l2:.9e}", f"old_vs_raycast_max_abs={max_abs:.9e}"]
    print(" ".join(fields), flush=True)


if __name__ == "__main__":
    main()
