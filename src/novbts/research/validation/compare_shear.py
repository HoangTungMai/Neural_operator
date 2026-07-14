#!/usr/bin/env python3
"""
Compare COARSE vs FINE FEM shear ground truth.

The fine mesh's payoff is that the tangential field is resolved well enough to
read the STICK CORE — so we can extract a stick radius c from the FEM field and
test it against Cattaneo-Mindlin c = a(1-g)^(1/3) (a = sqrt(R d)).  The coarse
mesh (~5 nodes across contact) cannot support this; the fine mesh (~10) can.

For each dataset we report:
  * resolution (markers across contact, distinct field values)
  * tangential saturation signal corr(peak_tang, travel)
  * direction alignment per slip mode
  * stick-radius extraction vs Cattaneo (the new quantitative check)
"""
import argparse
import numpy as np

from novbts.paths import FEM

MODE = ["normal", "stick", "partial_slip", "full_slip"]
G_STICK, G_PARTIAL, G_FULL = 0.04, 0.48, 1.0


def stick_radius_from_field(coords, disp_xy, cx, cy, drag, a):
    """Radial profile of along-drag displacement; stick radius c = where the
    profile drops to 50% of its core (r<a/3) value.  Returns c or nan."""
    r = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    along = disp_xy @ drag                       # signed projection onto drag dir
    inside = r <= a * 1.2
    if inside.sum() < 6:
        return np.nan
    core = along[r <= a / 3.0]
    if core.size < 1:
        return np.nan
    core_val = np.median(core)
    if abs(core_val) < 1e-9:
        return np.nan
    # bin radially, find first bin where mean along-drag falls below 50% of core
    order = np.argsort(r[inside])
    rr, aa = r[inside][order], along[inside][order]
    nb = 6
    edges = np.linspace(0, a * 1.2, nb + 1)
    prof_r, prof_v = [], []
    for k in range(nb):
        m = (rr >= edges[k]) & (rr < edges[k + 1])
        if m.any():
            prof_r.append(0.5 * (edges[k] + edges[k + 1]))
            prof_v.append(np.mean(aa[m]) / core_val)
    prof_r, prof_v = np.array(prof_r), np.array(prof_v)
    below = np.where(prof_v < 0.5)[0]
    if below.size == 0:
        return a            # never drops -> full stick (c ~ a)
    return float(prof_r[below[0]])


def analyse(path, label):
    d = np.load(path, allow_pickle=True)
    p, coords, disp = d["params"], d["coords"], d["disp"]
    sx, sy, mu, depth, R = p[:, 4], p[:, 5], p[:, 6], p[:, 2], p[:, 3]
    travel = np.sqrt(sx ** 2 + sy ** 2)
    g = travel / (mu * 0.01)
    a = np.sqrt(R * depth)
    tang = np.linalg.norm(disp[:, :, :2], axis=2)
    peak_tang = tang.max(1)

    span = coords[:, 0].max() - coords[:, 0].min()
    side = int(np.sqrt(coords.shape[0]))
    spacing = span / (side - 1)
    n_across = 2 * a.mean() / spacing
    n_distinct = len(np.unique(np.round(disp[:, :, 2], 6)))  # rough field richness

    corr = float(np.corrcoef(peak_tang, travel)[0, 1])

    drag = np.stack([sx, sy], 1) / np.maximum(travel[:, None], 1e-9)
    # stick radius for partial-slip frames (where a finite stick core exists)
    psel = (g >= G_PARTIAL) & (g < G_FULL)
    c_fem, c_theory, gg = [], [], []
    for i in np.where(psel)[0]:
        c = stick_radius_from_field(coords, disp[i, :, :2], 0.0, 0.0, drag[i], a[i])
        if np.isfinite(c):
            c_fem.append(c)
            c_theory.append(a[i] * (1 - min(g[i], 1.0)) ** (1 / 3))
            gg.append(g[i])
    c_fem, c_theory = np.array(c_fem), np.array(c_theory)

    print(f"\n===== {label}  ({path}) =====")
    print(f"  frames={p.shape[0]}  markers={coords.shape[0]} (side {side})  "
          f"gel_span={span*1000:.0f}mm")
    print(f"  contact a~{a.mean()*1000:.1f}mm  marker spacing={spacing*1000:.2f}mm  "
          f"-> ~{n_across:.1f} markers across contact diameter")
    print(f"  distinct uz field values: {n_distinct} (field richness)")
    print(f"  tangential saturation: corr(peak_tang,travel)={corr:+.3f}  "
          f"peak_tang={peak_tang.mean()*1000:.2f}±{peak_tang.std()*1000:.2f}mm")
    if len(c_fem) >= 3:
        rel = np.abs(c_fem - c_theory) / np.maximum(c_theory, 1e-9)
        print(f"  STICK RADIUS c (n={len(c_fem)} partial-slip frames):")
        print(f"    FEM c mean={c_fem.mean()*1000:.2f}mm  Cattaneo c=a(1-g)^1/3 mean={c_theory.mean()*1000:.2f}mm")
        print(f"    rel err |c_fem - c_theory|/c_theory = {rel.mean()*100:.0f}%  "
              f"(corr={np.corrcoef(c_fem,c_theory)[0,1]:+.2f})")
        print(f"    -> stick core {'READABLE — quantitative Cattaneo check possible' if rel.mean()<0.5 else 'noisy (resolution still marginal)'}")
    else:
        print(f"  STICK RADIUS: not extractable (only {len(c_fem)} usable frames; "
              f"resolution too coarse)")
    return dict(label=label, n_across=n_across, corr=corr,
                c_n=len(c_fem),
                c_err=(np.abs(c_fem-c_theory)/np.maximum(c_theory,1e-9)).mean() if len(c_fem)>=3 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", default=str(FEM / "shear_coarse.npz"))
    ap.add_argument("--fine", default=str(FEM / "shear_fine.npz"))
    args = ap.parse_args()
    rc = analyse(args.coarse, "COARSE (default mesh, gel 100mm)")
    rf = analyse(args.fine, "FINE (hex-res 24, gel 50mm)")
    def cstr(r):
        return "n/a" if r["c_n"] < 3 else f"{r['c_err']*100:.0f}% err (n={r['c_n']})"
    print("\n================= VERDICT thô vs mịn =================")
    print(f"  markers across contact:  coarse {rc['n_across']:.1f}  ->  fine {rf['n_across']:.1f}")
    print(f"  stick-radius vs Cattaneo: coarse {cstr(rc)}   ->  fine {cstr(rf)}")
    print(f"  slip saturation corr:    coarse {rc['corr']:+.2f}  ->  fine {rf['corr']:+.2f}")


if __name__ == "__main__":
    main()
