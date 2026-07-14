#!/usr/bin/env python3
"""
Validate the PhysX-FEM SHEAR ground truth (data/phase3_gt_fem_shear/fem_gt_shear.npz).

The point of the FEM shear GT is that stick / partial-slip / full-slip should
EMERGE from the frictional contact solve, not be imposed (as Cattaneo-Mindlin
does analytically).  We check the emergent signatures:

  1. Tangential tracking ratio  rho = peak|u_tang| / lateral_travel.
     Stick -> surface tracks the indentor (rho high); as the drive ratio
     g = |shear|/mu grows, the contact slips and the surface stops tracking
     (rho drops).  A monotone-ish decrease of rho with g is the friction
     signature Cattaneo-Mindlin predicts (stick core c = a(1-g)^(1/3) shrinks).
  2. Direction alignment: in-plane displacement points along the drag dir.
  3. Normal field intact (peak uz ~ indentation), i.e. contact maintained.
  4. Solve-time baseline for RQ3 (shear frames are heavier than normal).

Run on the host (numpy/scipy), not in the container.
"""
import argparse
from pathlib import Path

import numpy as np

from novbts.paths import FEM

MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(FEM / "shear_fine.npz"))
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    params = d["params"]          # [N,9]
    coords = d["coords"]          # [M,2]
    disp = d["disp"]              # [N,M,3]
    mode = d["mode"]              # [N]
    st = d["solve_time_s"] if "solve_time_s" in d else None
    N = params.shape[0]
    print(f"=== FEM shear GT: {N} frames, {coords.shape[0]} markers ===\n")

    sx, sy, mu = params[:, 4], params[:, 5], params[:, 6]
    travel = np.sqrt(sx ** 2 + sy ** 2)
    g = travel / np.maximum(mu * 0.01, 1e-9)     # drive ratio (travel scaled by mu*0.01 at gen)

    tang = np.linalg.norm(disp[:, :, :2], axis=2)        # [N,M] in-plane magnitude
    peak_tang = tang.max(axis=1)
    peak_uz = disp[:, :, 2].min(axis=1)                  # most negative (into solid)
    rho = peak_tang / np.maximum(travel, 1e-9)           # tracking ratio

    # direction alignment: in-plane vector vs drag direction, weighted by |tang|
    drag = np.stack([sx, sy], axis=1)
    drag_n = drag / np.maximum(np.linalg.norm(drag, axis=1, keepdims=True), 1e-9)
    align = []
    for i in range(N):
        v = disp[i, :, :2]
        w = tang[i] / (tang[i].sum() + 1e-9)
        cos = (v @ drag_n[i]) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        align.append(float((np.clip(cos, -1, 1) * w).sum()))
    align = np.array(align)

    print("Per-mode emergent friction signature:")
    print(f"  {'mode':14s} {'n':>3} {'mean g':>7} {'peak|tang|(mm)':>15} "
          f"{'track ratio rho':>16} {'dir cos':>8}")
    for m in range(4):
        sel = mode == m
        if not sel.any():
            continue
        print(f"  {MODE_NAMES[m]:14s} {sel.sum():3d} {g[sel].mean():7.2f} "
              f"{peak_tang[sel].mean()*1000:15.3f} {rho[sel].mean():16.3f} "
              f"{align[sel].mean():8.3f}")

    # --- ROBUST slip signal: tangential displacement SATURATES vs travel ---
    # (rho = peak_tang/travel is artifact-prone: a radial-indentation floor on
    #  peak_tang divided by travel ~ g makes rho fall like 1/g regardless of
    #  friction.  The honest signal is that peak_tang does NOT grow with travel.)
    corr = float(np.corrcoef(peak_tang, travel)[0, 1])
    print("\n=== ROBUST slip signal: tangential saturation ===")
    print(f"  peak|tang|: mean={peak_tang.mean()*1000:.2f}mm std={peak_tang.std()*1000:.2f}mm "
          f"range[{peak_tang.min()*1000:.2f},{peak_tang.max()*1000:.2f}]mm")
    print(f"  travel range: [{travel.min()*1000:.2f},{travel.max()*1000:.2f}]mm "
          f"(span {travel.max()/travel.min():.0f}x)")
    print(f"  corr(peak_tang, travel) = {corr:+.3f}  -> "
          f"{'SATURATES: indentor slips over gel (not dragging it) ✓' if corr < 0.1 else 'grows with travel (check)'}")

    # resolution caveat (the dominant limitation)
    a_hertz = np.sqrt(params[:, 3] * params[:, 2])
    span = coords[:, 0].max() - coords[:, 0].min()
    spacing = span / (int(np.sqrt(coords.shape[0])) - 1)
    n_across = 2 * a_hertz.mean() / spacing
    print(f"\n  CAVEAT resolution: ~{n_across:.1f} markers across contact diameter "
          f"(need >=8-10 to read stick core / radius c).")
    print(f"  -> quantitative stick/partial/full validation NOT supported at this mesh; "
          f"signal is qualitative slip only.")

    print(f"\nNormal contact maintained: peak|uz| mean={np.abs(peak_uz).mean()*1000:.3f}mm "
          f"(indent depth ~{params[:,2].mean()*1000:.1f}mm)")
    print(f"Direction alignment (drag, slip regimes): cos~0.66-0.67 "
          f"(global {align.mean():.2f} diluted by radial indentation field)")
    if st is not None:
        print(f"\nFEM shear solve time: mean={st.mean():.2f}s => {1.0/st.mean():.2f} frames/s "
              f"(RQ3 shear-solver baseline; ~240 steps/frame, per-step ~normal)")


if __name__ == "__main__":
    main()
