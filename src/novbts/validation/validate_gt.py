#!/usr/bin/env python3
"""
Ground-truth validator (Giai doan D).

Given ANY ground-truth dataset (.npz with params/coords/disp/mode) — whether
from gt_hertz_mindlin or the Isaac Sim PhysX-FEM extractor — this checks that
the displacement FIELD reproduces the exact contact-mechanics invariants:

  * peak normal displacement |u_z|
  * Hertz contact radius a            (exact: a = sqrt(R d))
  * Cattaneo-Mindlin stick radius c   (exact: c = a (1 - Q/uP)^(1/3))

It estimates each invariant from the field (the same way you would for a black-
box FEM solver) and reports the relative error vs the closed form.  A FEM GT
that matches Hertz-Mindlin here is trustworthy (reviewer-proof), especially in
the slip annulus where PhysX friction is weakest.

Also runs a resolution convergence check (res16/res32/res64) on shared params.

Usage:
  python3 scripts/validate_gt.py --gt-file data/phase3_gt/test_id.npz
  python3 scripts/validate_gt.py --gt-file data/phase3_gt_fem/test_id.npz   # later
"""

import argparse
from pathlib import Path

import numpy as np

from novbts.groundtruth.hertz_mindlin import hertz_scalars, mindlin_stick_radius, MODE_NAMES
from novbts.paths import ANALYTIC


def estimate_contact_radius(uz_frame, coords, cx, cy, frac=0.5):
    """
    Contact radius from the Hertz exact relation u_z(a) = 0.5 * u_z(0):
    the normal surface displacement at the contact edge is exactly half the
    central peak.  So a ~ radius where |u_z| crosses frac=0.5 of peak.
    Estimated on a radial profile (binned) to be robust to the marker grid.
    """
    r = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    mag = np.abs(uz_frame)
    peak = mag.max()
    if peak < 1e-9:
        return 0.0, peak
    order = np.argsort(r)
    r_s, m_s = r[order], mag[order]
    # first radius (moving outward) where the profile falls below frac*peak
    below = np.where(m_s <= frac * peak)[0]
    return (float(r_s[below[0]]) if below.size else float(r_s[-1])), peak


def estimate_stick_radius(disp_xy, coords, cx, cy, a_est, shear_dir):
    """
    Stick radius from the DIRECTED tangential displacement.

    Projecting onto the shear direction cancels most of the (radial, outward)
    Hertz push, isolating the shear response: ~constant (= delta) in the stick
    core, tapering through the slip annulus.  Estimate c as the radius where the
    radially-binned directed profile drops below 70% of its central plateau.
    """
    r = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    directed = disp_xy @ shear_dir  # signed component along shear
    # radial bins within the contact patch
    nb = 16
    edges = np.linspace(0, max(a_est, 1e-3), nb + 1)
    prof, centers = [], []
    for k in range(nb):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if m.any():
            prof.append(np.mean(directed[m])); centers.append(0.5 * (edges[k] + edges[k + 1]))
    if len(prof) < 3:
        return 0.0
    prof, centers = np.array(prof), np.array(centers)
    plateau = np.mean(prof[:2])  # innermost bins = stick core (= delta)
    if abs(plateau) < 1e-9:
        return 0.0
    # stick radius = where the directed profile first departs the flat core
    # plateau (leaves the stick zone).  0.9 detects the knee robustly even when
    # c is close to a (large beta) where the annulus taper stays high.
    below = centers[prof <= 0.9 * plateau]
    return float(below.min()) if below.size else float(a_est)


def validate(gt_file, sample=400):
    d = np.load(gt_file, allow_pickle=True)
    params, coords, disp, mode = d["params"], d["coords"], d["disp"], d["mode"]
    n = min(sample, params.shape[0])
    idx = np.linspace(0, params.shape[0] - 1, n).astype(int)

    a_err, uz_vals, c_err = [], [], []
    for i in idx:
        p = params[i]
        cx, cy, depth, R, sx, sy, mu, E, geom = p
        a_exact, P, p0, C = hertz_scalars(np.array([depth]), np.array([R]), np.array([E]))
        a_exact = float(a_exact[0])
        uz = disp[i, :, 2]
        a_est, peak = estimate_contact_radius(uz, coords, cx, cy)
        if a_exact > 1e-3:
            a_err.append(abs(a_est - a_exact) / a_exact)
        uz_vals.append(peak)
        # stick radius only meaningful for partial slip
        if mode[i] == 2:
            g = np.hypot(sx, sy) / max(mu, 1e-6)
            c_exact = float(mindlin_stick_radius(np.array([a_exact]), np.array([g]))[0])
            smag = np.hypot(sx, sy)
            shear_dir = np.array([sx, sy]) / max(smag, 1e-9)
            c_est = estimate_stick_radius(disp[i, :, :2], coords, cx, cy, a_est, shear_dir)
            if c_exact > 1e-3:
                c_err.append(abs(c_est - c_exact) / c_exact)

    print(f"\n=== GT validation: {gt_file} ===")
    print(f"  frames checked: {n}   markers: {coords.shape[0]}")
    print(f"  contact radius a   : mean rel err = {np.mean(a_err)*100:.2f}%  "
          f"(median {np.median(a_err)*100:.2f}%)")
    print(f"  peak |u_z|         : mean = {np.mean(uz_vals):.4f}  "
          f"range [{np.min(uz_vals):.4f}, {np.max(uz_vals):.4f}]")
    if c_err:
        print(f"  stick radius c     : mean rel err = {np.mean(c_err)*100:.2f}%  "
              f"(median {np.median(c_err)*100:.2f}%, n={len(c_err)})")
    counts = {MODE_NAMES[i]: int((mode == i).sum()) for i in range(4)}
    print(f"  mode balance       : {counts}")
    return {"a_rel_err": float(np.mean(a_err)),
            "c_rel_err": float(np.mean(c_err)) if c_err else None,
            "peak_uz_mean": float(np.mean(uz_vals))}


def convergence(data_dir):
    """Check peak |u_z| & contact radius converge with marker resolution."""
    data_dir = Path(data_dir)
    files = {16: data_dir / "test_ood_res16.npz",
             32: data_dir / "test_id.npz",
             64: data_dir / "test_ood_res64.npz"}
    if not all(f.exists() for f in files.values()):
        print("\n[convergence] resolution splits not all present, skipping.")
        return {}
    print("\n=== Resolution convergence (shared params where available) ===")
    out = {}
    for res, f in files.items():
        d = np.load(f, allow_pickle=True)
        uz = d["disp"][:, :, 2]
        peak = np.abs(uz).max(axis=1).mean()
        out[res] = float(peak)
        print(f"  res {res:3d}: mean peak|u_z| = {peak:.5f}")
    if 64 in out and 32 in out:
        rel = abs(out[32] - out[64]) / out[64]
        print(f"  |32 vs 64| relative change = {rel*100:.2f}%  "
              f"({'converged' if rel < 0.05 else 'NOT converged (<5% target)'})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-file", default=str(ANALYTIC / "test_id.npz"))
    ap.add_argument("--data-dir", default=str(ANALYTIC))
    args = ap.parse_args()
    validate(args.gt_file)
    convergence(args.data_dir)


if __name__ == "__main__":
    main()
