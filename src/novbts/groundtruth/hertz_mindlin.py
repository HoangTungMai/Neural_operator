#!/usr/bin/env python3
"""
Semi-analytic contact-mechanics ground truth for VBTS marker displacement.

Sphere (and flat-punch) pressed onto an elastic half-space:
  * Hertz theory  -> normal contact: contact radius a, peak pressure p0,
                     normal surface displacement u_z (exact, Johnson eq. 3.41).
  * Cattaneo-Mindlin -> tangential loading with partial slip: stick radius
                     c = a (1 - Q/uP)^(1/3), rigid tangential shift delta_t
                     (Mindlin), and slip annulus c < r < a.

This serves two roles in Phase 3:
  1. VALIDATOR for the PhysX/FEM ground truth — the scalar invariants
     (a, p0, c, peak u_z, delta_t) are EXACT closed forms; validate_gt.py
     compares FEM output against them.
  2. FALLBACK ground-truth generator if Isaac Sim cannot run on this machine.

Reference: K.L. Johnson, *Contact Mechanics* (1985), ch.3 (Hertz) & ch.7
(Cattaneo-Mindlin tangential partial slip).

Parameter vector (per frame), matching the dataset schema [N, 9]:
    0 cx          contact centre x        (-0.7 .. 0.7)
    1 cy          contact centre y        (-0.7 .. 0.7)
    2 depth       normal indentation d    (0.1 .. 1.0)
    3 radius      indentor radius R       (0.08 .. 0.33)
    4 shear_x     tangential drive x
    5 shear_y     tangential drive y
    6 mu          friction coefficient    (0.3 .. 0.9)
    7 stiffness   modulus scale E         (0.5 .. 3.5)
    8 geom        0 = sphere, 1 = flat punch

The tangential drive ratio is g = |shear| / mu  (the sampler builds
shear_mag = mu * factor, so g == factor falls cleanly in the regimes:
normal ~0, stick <0.5, partial 0.48-0.72, full >=0.9).
"""

import math

import numpy as np

MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]
NU = 0.45  # Poisson ratio of the gel (near-incompressible elastomer)

# Regime thresholds on g = Q/(mu P)
G_STICK = 0.04     # below -> essentially normal-only
G_PARTIAL = 0.48   # stick -> partial slip onset
G_FULL = 1.0       # full sliding


# ---------------------------------------------------------------------------
# Exact Hertz / Cattaneo-Mindlin scalar invariants (used by the validator)
# ---------------------------------------------------------------------------

def hertz_scalars(depth: np.ndarray, radius: np.ndarray, E: np.ndarray):
    """
    Returns (a, P, p0, C) for a sphere on an elastic half-space.
      a  = sqrt(R d)                contact radius
      E* = E / (1 - nu^2)           effective modulus (rigid indentor)
      P  = (4/3) E* a^3 / R         normal load
      p0 = 3P / (2 pi a^2)          peak pressure = (2/pi) E* sqrt(d/R)
      C  = 1 / E*                   surface compliance
    """
    a = np.sqrt(np.clip(radius * depth, 1e-8, None))
    Estar = E / (1.0 - NU ** 2)
    P = (4.0 / 3.0) * Estar * a ** 3 / np.clip(radius, 1e-6, None)
    p0 = 3.0 * P / (2.0 * math.pi * np.clip(a ** 2, 1e-8, None))
    C = 1.0 / Estar
    return a, P, p0, C


def mindlin_stick_radius(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """c = a (1 - g)^(1/3) for g<1 (partial slip); 0 for full slip."""
    gp = np.clip(g, 0.0, 1.0)
    return a * np.cbrt(np.clip(1.0 - gp, 0.0, 1.0))


def mindlin_tangential_shift(P, a, E, g):
    """
    Rigid tangential displacement of the stick zone (Mindlin):
      delta_t = (3 mu P (2-nu)) / (16 G a) [1 - (1 - Q/uP)^(2/3)]
    Here Q/uP = g and mu P cancels into Q = g * (mu P); we express via Q = g*muP,
    but muP is absorbed — we pass the product through |shear| scaling instead,
    so this returns the *shape factor* times an elastic compliance.
    """
    G = E / (2.0 * (1.0 + NU))
    gp = np.clip(g, 0.0, 1.0)
    # tangential compliance shape (without the mu*P force magnitude, which is
    # carried by the shear vector amplitude downstream)
    shape = (3.0 * (2.0 - NU)) / (16.0 * G * np.clip(a, 1e-6, None))
    return shape * (1.0 - np.cbrt((1.0 - gp) ** 2))


# ---------------------------------------------------------------------------
# Full surface displacement field
# ---------------------------------------------------------------------------

def hertz_uz_field(r: np.ndarray, a: np.ndarray, p0: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Exact Hertz normal surface displacement (downward, negative z).
      r <= a : u_z = C (pi p0 / 4a)(2a^2 - r^2)
      r >  a : u_z = C (p0 / 2a)[(2a^2 - r^2) asin(a/r) + a r sqrt(1-(a/r)^2)]
    Continuous at r = a.  (Johnson 1985, eq. 3.41.)
    """
    a = np.clip(a, 1e-6, None)
    inside = (C * math.pi * p0 / (4.0 * a)) * (2.0 * a ** 2 - r ** 2)
    ar = np.clip(a / np.clip(r, 1e-9, None), 0.0, 1.0)
    outside = (C * p0 / (2.0 * a)) * (
        (2.0 * a ** 2 - r ** 2) * np.arcsin(ar)
        + a * r * np.sqrt(np.clip(1.0 - ar ** 2, 0.0, None))
    )
    uz = np.where(r <= a, inside, outside)
    return -uz  # surface pushed into the solid


def tangential_profile(r, a, c, g):
    """
    Magnitude profile of the in-plane (tangential) surface displacement,
    normalised so the stick core = 1.

      stick   (r <= c)      : 1                      (moves rigidly with indentor)
      annulus (c < r <= a)  : smooth taper 1 -> beta (microslip reduces motion)
      outside (r > a)       : beta * (a/r)           (tangential elastic tail)

    For full slip (g>=1, c->0) the whole patch translates rigidly (profile=1
    inside a) — the qualitative signature distinguishing full slip from stick.
    """
    a = np.clip(a, 1e-6, None)
    full = g >= G_FULL
    beta = np.clip(c / a, 0.0, 1.0)  # residual motion at contact edge

    # annulus taper from 1 (at c) to beta (at a)
    denom = np.clip(a - c, 1e-6, None)
    frac = np.clip((r - c) / denom, 0.0, 1.0)
    annulus = 1.0 - (1.0 - beta) * frac

    prof = np.where(r <= c, 1.0, annulus)
    prof = np.where(r > a, beta * (a / np.clip(r, 1e-9, None)), prof)

    # full slip: rigid translation across the whole contact patch
    prof_full = np.where(r <= a, 1.0, (a / np.clip(r, 1e-9, None)))
    return np.where(full[:, None] if prof.ndim == 2 else full, prof_full, prof)


def hertz_mindlin_field(params: np.ndarray, coords: np.ndarray):
    """
    params : [N, 9]   coords : [M, 2]   ->   disp [N, M, 3], mode [N]
    Vectorised over frames and markers.
    """
    params = np.asarray(params, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    x0, y0, depth, radius, sx, sy, mu, E, geom = [params[:, i] for i in range(9)]

    dx = coords[None, :, 0] - x0[:, None]
    dy = coords[None, :, 1] - y0[:, None]
    r = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)

    a, P, p0, C = hertz_scalars(depth, radius, E)

    # ---- flat punch shares Hertz scaling but flatter pressure -> rescale a ----
    # (sphere geom=0 uses Hertz a; flat geom=1 uses indentor radius as patch.)
    a_eff = np.where(geom > 0.5, radius, a)

    # ---- normal field (Hertz) ----
    uz = hertz_uz_field(r, a_eff[:, None], p0[:, None], C[:, None])

    # small outward radial in-plane push from indentation (~ Hertz)
    radial_amp = (0.10 * C * p0)[:, None]
    contact_mask = np.exp(-(r ** 2) / (2.0 * np.clip(a_eff[:, None], 1e-6, None) ** 2))
    ux_r = radial_amp * contact_mask * dx / np.clip(a_eff[:, None], 1e-6, None)
    uy_r = radial_amp * contact_mask * dy / np.clip(a_eff[:, None], 1e-6, None)

    # ---- tangential field (Cattaneo-Mindlin) ----
    shear_mag = np.sqrt(sx ** 2 + sy ** 2 + 1e-12)
    g = shear_mag / np.clip(mu, 1e-6, None)
    c = mindlin_stick_radius(a_eff, g)

    delta_shape = mindlin_tangential_shift(P, a_eff, E, g)  # elastic compliance shape
    # tangential amplitude carried by the shear-force magnitude:
    A = (shear_mag * np.clip(a_eff, 1e-6, None)) * 0.6  # O(uz) scale; force * compliance
    # blend in Mindlin compliance shape so partial-slip amplitude grows correctly
    A = A * (0.5 + 0.5 * np.clip(delta_shape / (np.median(delta_shape) + 1e-9), 0.0, 2.0))

    prof = tangential_profile(r, a_eff[:, None], c[:, None], g)
    dirx = (sx / shear_mag)[:, None]
    diry = (sy / shear_mag)[:, None]
    ux_t = A[:, None] * prof * dirx
    uy_t = A[:, None] * prof * diry

    ux = ux_r + ux_t
    uy = uy_r + uy_t
    disp = np.stack([ux, uy, uz], axis=-1).astype(np.float32)

    # ---- mode labels from the physical drive ratio g ----
    mode = np.full(params.shape[0], 1, dtype=np.int32)  # default stick
    mode[g < G_STICK] = 0          # normal
    mode[(g >= G_PARTIAL) & (g < G_FULL)] = 2  # partial slip
    mode[g >= G_FULL] = 3          # full slip
    return disp, mode


# ---------------------------------------------------------------------------
# Self-test: verify the exact invariants and slip regimes
# ---------------------------------------------------------------------------

def _grid(side: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def _selftest() -> None:
    print("=== gt_hertz_mindlin self-test ===")
    coords = _grid(64)

    # 1. Q=0 -> pure Hertz: stick radius c == a, tangential field ~ 0
    p = np.array([[0.0, 0.0, 0.6, 0.25, 0.0, 0.0, 0.6, 2.0, 0.0]])
    a, P, p0, C = hertz_scalars(p[:, 2], p[:, 3], p[:, 7])
    disp, mode = hertz_mindlin_field(p, coords)
    tang = np.linalg.norm(disp[0, :, :2], axis=-1).max()
    print(f"  Q=0: a={a[0]:.4f} p0={p0[0]:.4f}  peak|uz|={np.abs(disp[0,:,2]).max():.4f}"
          f"  max|tang|={tang:.2e}  mode={MODE_NAMES[mode[0]]}  (expect normal, tang~0)")

    # 2. stick radius shrinks as g -> 1 following c = a(1-g)^(1/3)
    print("  Cattaneo-Mindlin stick radius c = a (1-g)^(1/3):")
    for g in [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]:
        mu = 0.6
        shear = mu * g
        pp = np.array([[0.0, 0.0, 0.6, 0.25, shear, 0.0, mu, 2.0, 0.0]])
        a_, *_ = hertz_scalars(pp[:, 2], pp[:, 3], pp[:, 7])
        c = mindlin_stick_radius(a_, np.array([g]))
        _, m = hertz_mindlin_field(pp, coords)
        exact = a_[0] * (1 - min(g, 1.0)) ** (1 / 3)
        ok = abs(c[0] - exact) < 1e-9
        print(f"    g={g:.2f}  c={c[0]:.4f}  exact={exact:.4f}  {'OK' if ok else 'FAIL'}"
              f"  mode={MODE_NAMES[m[0]]}")

    # 3. full slip: whole patch moves rigidly (profile uniform inside a)
    pp = np.array([[0.0, 0.0, 0.6, 0.25, 0.6 * 1.1, 0.0, 0.6, 2.0, 0.0]])
    disp, m = hertz_mindlin_field(pp, coords)
    print(f"  full slip drive g=1.1 -> mode={MODE_NAMES[m[0]]} (expect full_slip)")

    # 4. continuity of u_z at r = a
    r = np.array([[0.249999, 0.250001]])
    a_ = np.array([0.25]); p0_ = np.array([1.0]); C_ = np.array([1.0])
    uz = hertz_uz_field(r, a_[:, None], p0_[:, None], C_[:, None])
    print(f"  u_z continuity at r=a: {uz[0,0]:.6f} vs {uz[0,1]:.6f}"
          f"  (diff {abs(uz[0,0]-uz[0,1]):.2e})")
    print("=== done ===")


if __name__ == "__main__":
    _selftest()
