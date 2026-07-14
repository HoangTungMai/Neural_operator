#!/usr/bin/env python3
"""Analytic-free mesh imprint profiles for geometry-OOD inputs.

The stored ``contact_profile`` is a geometric penetration height field on the
marker grid.  It is computed from the indenter surface mesh and pose only; gel
displacement is never used.
"""
from __future__ import annotations

import numpy as np


def icosphere_surface(radius: float, subdiv: int = 2, center=(0.0, 0.0, 0.0)):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    faces = [list(f) for f in faces]

    def midpoint(cache, a, b, vlist):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = (vlist[a] + vlist[b]) / 2.0
        vlist.append(m)
        idx = len(vlist) - 1
        cache[key] = idx
        return idx

    vlist = [v for v in verts]
    for _ in range(int(subdiv)):
        cache = {}
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(cache, a, b, vlist)
            bc = midpoint(cache, b, c, vlist)
            ca = midpoint(cache, c, a, vlist)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    V = np.asarray(vlist, dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * float(radius)
    V = V + np.asarray(center, dtype=np.float64)
    return V, np.asarray(faces, dtype=np.uint32)


def ellipsoid_surface(rx: float, ry: float, rz: float, subdiv: int = 2, center=(0.0, 0.0, 0.0)):
    V, F = icosphere_surface(1.0, subdiv=subdiv, center=(0.0, 0.0, 0.0))
    V = V * np.array([rx, ry, rz], dtype=np.float64) + np.asarray(center, dtype=np.float64)
    return V, F


def _bolt_hex_xy(radius: float, points_per_corner: int = 4):
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False) + np.pi / 6.0
    verts = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
    cut = 0.28
    pts = []
    for i, v in enumerate(verts):
        prev_v = verts[(i - 1) % len(verts)]
        next_v = verts[(i + 1) % len(verts)]
        a = (1.0 - cut) * v + cut * prev_v
        b = (1.0 - cut) * v + cut * next_v
        for t in np.linspace(0.0, 1.0, points_per_corner, endpoint=False):
            q = (1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * v + t ** 2 * b
            pts.append(q)
    return np.asarray(pts, dtype=np.float64)


def bolt_hex_surface(radius: float, half_z: float, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    xy = _bolt_hex_xy(radius, points_per_corner=4)
    verts = []
    for z in (-half_z, half_z):
        for x, y in xy:
            verts.append((cx + x, cy + y, cz + z))
    bottom_ci = len(verts)
    verts.append((cx, cy, cz - half_z))
    top_ci = len(verts)
    verts.append((cx, cy, cz + half_z))
    n = len(xy)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
        faces.append((bottom_ci, j, i))
        faces.append((top_ci, n + i, n + j))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def cone_surface(radius: float, half_z: float, n_theta: int = 64, center=(0.0, 0.0, 0.0)):
    """Closed circular cone with a sharp lower tip and a shallow conical face."""
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    rim_z = cz - 0.55 * half_z
    rim = [(cx + radius * np.cos(th), cy + radius * np.sin(th), rim_z) for th in theta]
    tip_i = len(rim)
    verts = list(rim)
    verts.append((cx, cy, cz - half_z))
    top_i = len(verts)
    verts.append((cx, cy, cz + half_z))

    faces = []
    nt = int(n_theta)
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((tip_i, i, j))
        faces.append((i, n_theta + 2 + j, j))
        faces.append((i, n_theta + 2 + i, n_theta + 2 + j))
    top_start = len(verts)
    for x, y, _ in rim:
        verts.append((x, y, cz + half_z))
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((top_i, j, i))
        faces[-1] = (top_i, top_start + i, top_start + j)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def multi_lobe_surface(radius: float, half_z: float, *, lobes: int = 5,
                       amp: float = 0.22, n_theta: int = 96, n_r: int = 10,
                       center=(0.0, 0.0, 0.0)):
    """Closed star-shaped multi-lobe tool with separated shallow contact peaks.

    The XY footprint is a radial graph, r(theta)=R(1+a cos(k theta)), keeping the
    fan-tet path deterministic.  The lower surface has k downward petal tips; at
    shallow indentation the raycast profile activates as several disjoint patches.
    """
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    nt = int(n_theta)
    nr = int(n_r)
    k = int(lobes)
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    boundary_r = radius * (1.0 + float(amp) * np.cos(k * theta))

    def ring_xy(ir):
        frac = ir / nr
        rr = frac * boundary_r
        return cx + rr * np.cos(theta), cy + rr * np.sin(theta), frac

    def bottom_z(frac, th):
        # Petal minima sit on a ring. Valleys and the centre are higher, so a
        # small press produces multi-contact rather than one filled footprint.
        angular_peak = 0.5 * (1.0 + np.cos(k * th))
        radial_peak = np.exp(-((frac - 0.62) / 0.24) ** 2)
        peak = angular_peak ** 2.0 * radial_peak
        lift = 0.46 * half_z * (1.0 - peak)
        return cz - half_z + lift

    verts = []
    bottom_center = 0
    verts.append((cx, cy, float(bottom_z(0.0, 0.0))))
    bottom_rings = []
    for ir in range(1, nr + 1):
        xs, ys, frac = ring_xy(ir)
        z = bottom_z(frac, theta)
        inds = []
        for i in range(nt):
            inds.append(len(verts))
            verts.append((float(xs[i]), float(ys[i]), float(z[i])))
        bottom_rings.append(inds)

    top_center = len(verts)
    verts.append((cx, cy, cz + half_z))
    top_rings = []
    for ir in range(1, nr + 1):
        xs, ys, _ = ring_xy(ir)
        inds = []
        for i in range(nt):
            inds.append(len(verts))
            verts.append((float(xs[i]), float(ys[i]), cz + half_z))
        top_rings.append(inds)

    faces = []
    first = bottom_rings[0]
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((bottom_center, first[j], first[i]))
    for a, b in zip(bottom_rings[:-1], bottom_rings[1:]):
        for i in range(nt):
            j = (i + 1) % nt
            faces.append((a[i], b[j], a[j]))
            faces.append((a[i], b[i], b[j]))

    first = top_rings[0]
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((top_center, first[i], first[j]))
    for a, b in zip(top_rings[:-1], top_rings[1:]):
        for i in range(nt):
            j = (i + 1) % nt
            faces.append((a[i], a[j], b[j]))
            faces.append((a[i], b[j], b[i]))

    bot = bottom_rings[-1]
    top = top_rings[-1]
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((bot[i], top[j], bot[j]))
        faces.append((bot[i], top[i], top[j]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def radial_prism_surface(radius: float, half_z: float, *, teeth: int = 12,
                         amp: float = 0.16, n_theta: int = 144,
                         center=(0.0, 0.0, 0.0)):
    """Closed flat-bottom star-shaped prism for gear/knurled-cap held-out tests."""
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    # Smooth deterministic teeth keep the radial graph star-shaped and fan-tet valid.
    rr = radius * (1.0 + float(amp) * np.cos(int(teeth) * theta))
    xy = np.stack([cx + rr * np.cos(theta), cy + rr * np.sin(theta)], axis=-1)
    verts = []
    for z in (-half_z, half_z):
        for x, y in xy:
            verts.append((float(x), float(y), cz + z))
    bottom_ci = len(verts)
    verts.append((cx, cy, cz - half_z))
    top_ci = len(verts)
    verts.append((cx, cy, cz + half_z))
    faces = []
    nt = int(n_theta)
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((i, j, nt + j))
        faces.append((i, nt + j, nt + i))
        faces.append((bottom_ci, j, i))
        faces.append((top_ci, nt + i, nt + j))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def rounded_die_surface(radius: float, half_z: float, subdiv: int = 3, center=(0.0, 0.0, 0.0)):
    """Superellipsoid-like rounded cube/die, convex and star-shaped."""
    V, F = icosphere_surface(1.0, subdiv=subdiv, center=(0.0, 0.0, 0.0))
    p = 0.34
    V = np.sign(V) * np.power(np.abs(V), p)
    V = V / np.max(np.abs(V), axis=0, keepdims=True)
    V = V * np.array([radius, radius, half_z], dtype=np.float64)
    V = V + np.asarray(center, dtype=np.float64)
    return V, F


def pebble_surface(radius: float, half_z: float, subdiv: int = 3, center=(0.0, 0.0, 0.0)):
    """Low-frequency asymmetric organic pebble; deterministic and star-shaped."""
    V, F = icosphere_surface(1.0, subdiv=subdiv, center=(0.0, 0.0, 0.0))
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    theta = np.arctan2(y, x)
    radial = 1.0 + 0.08 * np.sin(2.0 * theta + 0.7) + 0.05 * z + 0.035 * np.cos(3.0 * theta - 1.2) * (1.0 - z * z)
    radial = np.clip(radial, 0.82, 1.18)
    V = V * radial[:, None]
    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    V[:, 2] = 2.0 * (V[:, 2] - zmin) / (zmax - zmin) - 1.0
    V = V * np.array([radius, 0.88 * radius, half_z], dtype=np.float64)
    V = V + np.asarray(center, dtype=np.float64)
    return V, F


def mesh_object_surface(name: str, radius: float, half_z: float, r2: float | None = None,
                        subdiv: int = 2):
    """Return local-centred (V, F, bottom_off) for a deterministic Tier-4 object."""
    obj = (name or "bolt_hex").strip()
    if obj in ("bolt_hex", "rounded_hex_bolt"):
        V, F = bolt_hex_surface(radius, half_z, center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    if obj in ("rounded_tip", "curved_tip", "mesh_ellipsoid"):
        ry = float(radius if r2 is None or r2 <= 0 else r2)
        V, F = ellipsoid_surface(radius, ry, half_z, subdiv=subdiv, center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    if obj in ("sphere", "sphere_mesh", "curved_sphere"):
        V, F = icosphere_surface(radius, subdiv=subdiv, center=(0.0, 0.0, 0.0))
        return V, F, float(radius)
    if obj in ("cone", "sharp_cone", "point_tip"):
        V, F = cone_surface(radius, half_z, center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    if obj in ("multi_lobe", "multilobe", "flower_lobe"):
        V, F = multi_lobe_surface(radius, half_z, center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    if obj in ("rounded_die", "beveled_cube", "rounded_cube"):
        V, F = rounded_die_surface(radius, half_z, subdiv=max(2, subdiv), center=(0.0, 0.0, 0.0))
        return V, F, float(-np.min(V[:, 2]))
    if obj in ("gear", "knurled_cap"):
        V, F = radial_prism_surface(radius, half_z, center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    if obj in ("pebble", "organic_pebble"):
        V, F = pebble_surface(radius, half_z, subdiv=max(2, subdiv), center=(0.0, 0.0, 0.0))
        return V, F, float(half_z)
    raise ValueError(f"unsupported Tier-4 mesh object {name!r}")


def raycast_bottom_profile(coords: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                           *, center_xy=(0.0, 0.0), center_z: float = 0.0,
                           eps: float = 1e-12) -> np.ndarray:
    """Vertical raycast lower surface z, returned as ``clip(-z_bottom, 0)``.

    ``vertices`` are local object vertices.  ``center_xy`` and ``center_z`` place
    the object in gel-top coordinates, where z=0 is the undeformed gel surface.
    """
    pts = np.asarray(coords, dtype=np.float64)
    V = np.asarray(vertices, dtype=np.float64).copy()
    V[:, 0] += float(center_xy[0])
    V[:, 1] += float(center_xy[1])
    V[:, 2] += float(center_z)
    F = np.asarray(faces, dtype=np.int64)
    z_bottom = np.full(pts.shape[0], np.inf, dtype=np.float64)
    px = pts[:, 0]
    py = pts[:, 1]

    for tri in F:
        a, b, c = V[tri]
        x0, y0 = a[:2]
        x1, y1 = b[:2]
        x2, y2 = c[:2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) <= eps:
            continue
        xmin = min(x0, x1, x2) - eps
        xmax = max(x0, x1, x2) + eps
        ymin = min(y0, y1, y2) - eps
        ymax = max(y0, y1, y2) + eps
        cand = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
        if not np.any(cand):
            continue
        x = px[cand]
        y = py[cand]
        w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den
        w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not np.any(inside):
            continue
        idx = np.flatnonzero(cand)[inside]
        z = w0[inside] * a[2] + w1[inside] * b[2] + w2[inside] * c[2]
        z_bottom[idx] = np.minimum(z_bottom[idx], z)

    profile = np.where(np.isfinite(z_bottom), np.clip(-z_bottom, 0.0, None), 0.0)
    return profile.astype(np.float32)


def mesh_contact_profile(coords: np.ndarray, params_row: np.ndarray, object_name: str,
                         *, half_z: float, r2: float | None = None, subdiv: int = 2) -> np.ndarray:
    p = np.asarray(params_row, dtype=np.float64).reshape(-1)
    cx, cy, depth, radius = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    ry = float(p[9]) if p.shape[0] > 9 else r2
    if r2 is not None:
        ry = float(r2)
    V, F, bottom_off = mesh_object_surface(object_name, radius, half_z, ry, subdiv=subdiv)
    center_z = bottom_off - depth
    flat = raycast_bottom_profile(coords, V, F, center_xy=(cx, cy), center_z=center_z)
    side = int(round(np.sqrt(flat.shape[0])))
    return flat.reshape(side, side)


def sphere_parabolic_penetration(coords: np.ndarray, params_row: np.ndarray) -> np.ndarray:
    p = np.asarray(params_row, dtype=np.float64).reshape(-1)
    cx, cy, depth, radius = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    dx = np.asarray(coords)[:, 0] - cx
    dy = np.asarray(coords)[:, 1] - cy
    pen = np.clip(depth - (dx * dx + dy * dy) / (2.0 * radius), 0.0, None)
    side = int(round(np.sqrt(pen.shape[0])))
    return pen.astype(np.float32).reshape(side, side)
