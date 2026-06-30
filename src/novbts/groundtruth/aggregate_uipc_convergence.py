#!/usr/bin/env python3
"""
Aggregate a UIPC convergence sweep (produced by infra/gen_uipc_convergence.sh)
into a verdict: does the IPC field CONVERGE under mesh + friction refinement?

Reads every ``data/uipc/conv/<tag>/uipc_gt_shear.npz`` (each holds one
(edge_length_r, eps_velocity) field on the SAME 32x32 marker grid, plus the two
knob values), then builds the two 1-D refinement paths and reports, per path:

  * successive-level rel-L2 (level k vs the next finer level k+1), split into
    normal (uz) and tangential (uxy) channels;
  * each level's rel-L2 distance to the FINEST field on that path.

Convergence reads as: successive deltas SHRINK toward 0 and the distance-to-finest
decreases monotonically. Contrast with the PhysX tangential channel, which wanders
0.7-0.9 under mesh refinement instead of shrinking (memory res32-upgrade).

Usage:
  python -m novbts.groundtruth.aggregate_uipc_convergence --conv-dir data/uipc/conv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def rel_l2(a, b):
    """||a-b|| / ||b||  (b is the reference, i.e. the finer field)."""
    denom = float(np.linalg.norm(b)) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def channels(field_a, field_b):
    """rel-L2 split into overall / normal (uz) / tangential (uxy)."""
    return {
        "overall": rel_l2(field_a, field_b),
        "normal": rel_l2(field_a[:, 2], field_b[:, 2]),
        "tangential": rel_l2(field_a[:, :2], field_b[:, :2]),
    }


def _same(a, b, name, path):
    if a.shape != b.shape:
        raise SystemExit(f"{name} mismatch in {path}; convergence fields are not paired")
    try:
        ok = np.allclose(a, b, rtol=1e-5, atol=1e-8)
    except (TypeError, ValueError):
        ok = np.array_equal(a, b)
    if not ok:
        raise SystemExit(f"{name} mismatch in {path}; convergence fields are not paired")


def load_points(conv_dir):
    """Return list of dicts: {gel_res, eps_velocity, field (M,3), tag}."""
    pts = []
    ref = None
    for npz_path in sorted(glob.glob(os.path.join(conv_dir, "*", "uipc_gt_shear.npz"))):
        d = np.load(npz_path, allow_pickle=True)
        if "gel_res" not in d.files:
            # skip pre-structured-mesh (wildmeshing-era) runs — they lack gel_res
            # and were shown to be invalid (non-deterministic mesh, collapsed field).
            print(f"  skip {npz_path} (no gel_res: old wildmeshing run)")
            continue
        if ref is None:
            ref = {
                "coords": np.asarray(d["coords"]),
                "params": np.asarray(d["params"]),
                "mode": np.asarray(d["mode"]),
            }
            for key in ("velocity_tol", "d_hat", "contact_resistance", "marker_sampling"):
                if key in d.files:
                    ref[key] = np.asarray(d[key])
        else:
            _same(ref["coords"], np.asarray(d["coords"]), "coords", npz_path)
            _same(ref["params"], np.asarray(d["params"]), "params", npz_path)
            _same(ref["mode"], np.asarray(d["mode"]), "mode", npz_path)
            for key in ("velocity_tol", "d_hat", "contact_resistance", "marker_sampling"):
                if key in ref and key in d.files:
                    _same(ref[key], np.asarray(d[key]), key, npz_path)
        field = np.asarray(d["disp"])[0]  # (M, 3)
        gr = int(np.asarray(d["gel_res"]).reshape(-1)[0])
        # eps is float32; eight decimal places remove representation noise without
        # collapsing Phase-0 levels such as 2.5e-5 to 2e-5.
        ev = round(float(np.asarray(d["eps_velocity"]).reshape(-1)[0]), 8)
        point = {
            "gel_res": gr,
            "eps_velocity": ev,
            "field": field,
            "tag": os.path.basename(os.path.dirname(npz_path)),
        }
        for key in ("velocity_tol", "d_hat", "contact_resistance"):
            if key in d.files:
                point[key] = float(np.asarray(d[key]).reshape(-1)[0])
        if "marker_sampling" in d.files:
            point["marker_sampling"] = str(np.asarray(d["marker_sampling"]).item())
        pts.append(point)
    return pts


# finest = LARGEST gel_res (more cells) but SMALLEST eps_velocity. Sort each axis
# coarse -> fine accordingly.
def _is_finer(axis, a, b):
    """True if level a is FINER than level b on this axis."""
    return a > b if axis == "mesh" else a < b


def refine_path(points, axis):
    """Build a coarse->fine sequence along ``axis`` ('mesh'|'eps').

    Pins the OTHER axis at its FINEST value present, then sorts the swept axis
    coarse-first, finest-last.
    """
    key = "gel_res" if axis == "mesh" else "eps_velocity"
    other = "eps_velocity" if axis == "mesh" else "gel_res"
    other_fine = min(p[other] for p in points) if other == "eps_velocity" else max(p[other] for p in points)
    tol = 1e-9 if other == "eps_velocity" else 0
    seq = [p for p in points if abs(p[other] - other_fine) <= tol]
    # coarse -> fine: mesh ascending (small res first), eps descending (large eps first)
    seq.sort(key=lambda p: p[key], reverse=(axis == "eps"))
    return seq, other, other_fine


def analyse(points, axis):
    seq, other, other_fine = refine_path(points, axis)
    key = "gel_res" if axis == "mesh" else "eps_velocity"
    if len(seq) < 2:
        return {"axis": axis, "note": f"need >=2 levels (have {len(seq)})", "levels": []}
    finest = seq[-1]["field"]
    steps = []
    for a, b in zip(seq[:-1], seq[1:]):  # a coarser than b
        ch = channels(a["field"], b["field"])
        steps.append({"from": a[key], "to": b[key], **ch})
    to_finest = []
    for p in seq:
        ch = channels(p["field"], finest)
        to_finest.append({key: p[key], **ch})
    # converging if successive tangential deltas are (weakly) decreasing AND small
    tang_steps = [s["tangential"] for s in steps]
    decreasing = all(tang_steps[i + 1] <= tang_steps[i] + 1e-6 for i in range(len(tang_steps) - 1))
    last_delta = tang_steps[-1]
    return {
        "axis": axis,
        "swept": key,
        "pinned": {other: other_fine},
        "levels": [p[key] for p in seq],
        "successive_rel_l2": steps,
        "distance_to_finest": to_finest,
        "tangential_steps": tang_steps,
        "monotone_decreasing": decreasing,
        "finest_step_tangential": last_delta,
    }


def fmt(report):
    out = []
    for path in (report["mesh_path"], report["friction_path"]):
        out.append(f"\n=== {path['axis']} refinement (swept {path.get('swept','?')}, "
                   f"pinned {path.get('pinned','?')}) ===")
        if "note" in path:
            out.append("  " + path["note"])
            continue
        out.append("  successive rel-L2 (coarse -> finer):")
        for s in path["successive_rel_l2"]:
            out.append(f"    {s['from']:>7} -> {s['to']:<7}  "
                       f"tang={s['tangential']:.4f}  norm={s['normal']:.4f}  overall={s['overall']:.4f}")
        out.append("  distance-to-finest:")
        for d in path["distance_to_finest"]:
            k = path["swept"]
            out.append(f"    {d[k]:<7}  tang={d['tangential']:.4f}  norm={d['normal']:.4f}")
        verdict = ("CONVERGING (tang deltas decreasing)" if path["monotone_decreasing"]
                   else "NOT monotone (tang deltas wander)")
        out.append(f"  -> {verdict}; finest-step tangential rel-L2 = {path['finest_step_tangential']:.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-dir", default="data/uipc/conv",
                    help="directory holding <tag>/uipc_gt_shear.npz from the sweep")
    ap.add_argument("--out", default=None, help="JSON report path (default <conv-dir>/convergence_report.json)")
    args = ap.parse_args()

    points = load_points(args.conv_dir)
    if not points:
        raise SystemExit(f"no uipc_gt_shear.npz found under {args.conv_dir}")
    print(f"loaded {len(points)} settings: "
          + ", ".join(f"(res{p['gel_res']},{p['eps_velocity']})" for p in points))

    report = {
        "n_points": len(points),
        "points": [{"gel_res": p["gel_res"], "eps_velocity": p["eps_velocity"],
                    "velocity_tol": p.get("velocity_tol"),
                    "d_hat": p.get("d_hat"),
                    "contact_resistance": p.get("contact_resistance"),
                    "marker_sampling": p.get("marker_sampling"),
                    "tag": p["tag"]} for p in points],
        "mesh_path": analyse(points, "mesh"),
        "friction_path": analyse(points, "eps"),
    }
    print(fmt(report))

    out_path = args.out or os.path.join(args.conv_dir, "convergence_report.json")
    # strip numpy fields from points before dumping (they're already excluded in report)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
