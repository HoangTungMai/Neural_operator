#!/usr/bin/env python3
"""
SHEAR ground-truth via UIPC (Unified Incremental Potential Contact) — TacEx backend.

WHY THIS EXISTS
---------------
The PhysX deformable GT (``isaac_extract_shear.py``) does NOT converge in the
tangential channel under mesh refinement: paired full-set distances are
res24<->res32 = 0.89, res32<->res40 = 0.70 (they wander instead of shrinking),
and PhysX blows up past ~res-48 so a converged tangential target is
unreachable. The normal channel is fine (Hertz contact-radius ~1.3%). The
culprit is a MODEL/solver error (position-based solver + regularised Coulomb
friction), not a discretisation error, so finer meshes cannot remove it.

IPC fixes exactly this: the smooth (lagged) friction model is mesh-independent
in the limit ``eps_velocity -> 0`` and the barrier contact is convergent under
mesh refinement (``gel_res -> infinity``). This driver reproduces ONE
indent+shear configuration with UIPC and runs a CONVERGENCE TEST over the two
knobs that make IPC converge where PhysX cannot:

  * ``gel_res``      — gel structured-mesh resolution (mesh refinement). A
                       DETERMINISTIC structured tet box, not wildmeshing: a
                       convergence study needs reproducible, monotone refinement
                       and a flat regular top face for well-posed field sampling.
  * ``eps_velocity`` — friction smoothing velocity [m/s] (friction model limit)

NOTE ON FRAMING: choosing IPC over PhysX is a FRAMEWORK step to obtain a
trustworthy GT, NOT a scientific contribution — TacEx already built GIPC to fix
PhysX. This convergence study only CALIBRATES the GT pipeline (pick a resolution,
quantify the GPU solver's run-to-run noise so we know to average K runs). The
paper's science remains the FNO operator; do not headline "PhysX vs IPC".

OUTPUT FORMAT
-------------
Matches ``isaac_extract_shear.py`` so the same benchmark / aggregate / paired-
comparison tooling works apples-to-apples against the PhysX sweeps:
``params, coords, disp, mode, solve_time_s, meta`` plus UIPC-specific provenance
fields (``gel_res, eps_velocity, d_hat, contact_resistance, n_tet_verts``) and
per-marker contact-traction fields ``contact_force_normal`` and
``contact_force_friction`` (N/m^2), plus exact pre-sampling nodal sums
``contact_force_total_normal`` and ``contact_force_total_friction`` (N).
All contact quantities act on the gel: a downward indentation therefore has
negative ``F_z``.
Coordinate convention is Z-up (gel top face at z=GEL_Z, uz is the normal
component) — identical to the PhysX driver, NOT the Y-up libuipc-samples.

HOW IT RUNS
-----------
Inside the ``isaac-lab-tacex:latest`` container, like the PhysX driver:

    /isaac-sim/python.sh -m novbts.research.groundtruth.tacex_uipc_extract_shear --smoke

Modes:
  --smoke         one indent+shear frame at the default (gel_res, eps_velocity),
                  verbose, no save, prints SMOKE_UIPC_OK on success.
  --single        run exactly ONE (gel_res, eps_velocity) setting and save the
                  field (for an external shell loop, à la infra/gen_fem_sweep*.sh).
  --convergence   (default for the full run) sweep --gel-res-list x
                  --eps-velocity-list in-process and write a convergence table.

Progress is logged line-by-line to /work/fem_progress_uipc.txt (Isaac swallows
stdout; watch that file, NOT stdout).
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
# --- run modes --------------------------------------------------------------
parser.add_argument("--smoke", action="store_true",
                    help="1 indent+shear frame, verbose, no save")
parser.add_argument("--single", action="store_true",
                    help="run ONE (gel_res, eps_velocity) and save the field")
parser.add_argument("--convergence", action="store_true",
                    help="sweep gel-res-list x eps-velocity-list in-process (default full run)")
parser.add_argument("--batch", action="store_true",
                    help="ONE Isaac boot, loop all frames x reps of a combo in-process (amortise ~23s "
                         "boot). Fixed combo (gel-res/eps/R/mu/youngs); per-frame depth/g/sx/sy from --batch-rows.")
parser.add_argument(
    "--batch-rows",
    default="",
    help=(
        "path to a whitespace table, one row per frame: "
        "'frame_idx depth g sx sy [load_mode [R mu youngs]]'"
    ),
)
parser.add_argument("--batch-reps", type=int, default=6, help="K replicate runs per frame in --batch")
# --- I/O --------------------------------------------------------------------
parser.add_argument("--out", default="/work/data/phase3_gt_uipc_shear")
parser.add_argument("--marker-side", type=int, default=32,
                    help="marker grid side (32 to match the res-32 FEM pipeline)")
# --- one indent+shear configuration (mirror isaac_extract_shear.py) ---------
parser.add_argument("--depth", type=float, default=0.005, help="normal indentation depth (m)")
parser.add_argument("--shear", type=float, default=0.004, help="lateral travel (m)")
parser.add_argument("--shear-x", type=float, default=None,
                    help="lateral x travel (m); overrides --shear for paired PhysX frames")
parser.add_argument("--shear-y", type=float, default=None,
                    help="lateral y travel (m); default 0 unless --shear-x is set")
parser.add_argument("--drive-ratio", type=float, default=None,
                    help="optional intended Cattaneo-Mindlin drive ratio g used only "
                         "to store a PhysX-compatible mode label for scripted sweeps")
parser.add_argument("--gel-xy", type=float, default=0.10, help="gel footprint x=y (m)")
parser.add_argument("--gel-z", type=float, default=0.04, help="gel thickness (m)")
parser.add_argument("--indentor-r", type=float, default=0.02, help="sphere indentor radius (m)")
parser.add_argument("--indentor-geom", choices=["sphere", "cylinder", "cuboid", "ellipsoid", "mesh"],
                    default="sphere", help="indentor contact geometry")
parser.add_argument("--indentor-half-z", type=float, default=-1.0,
                    help="indentor half-height / z semi-axis (m); default = --indentor-r")
parser.add_argument("--indentor-r2", type=float, default=-1.0,
                    help="ellipsoid y semi-axis (m); default = --indentor-r")
parser.add_argument("--indentor-mesh", default="", help="USD/OBJ mesh path for --indentor-geom mesh")
parser.add_argument("--mu", type=float, default=0.6, help="friction coeff (-> default_friction_ratio)")
parser.add_argument("--youngs", type=float, default=1.0e5, help="gel Young's modulus E (Pa)")
parser.add_argument("--poisson", type=float, default=0.45, help="gel Poisson ratio")
parser.add_argument("--gel-density", type=float, default=1.0e3, help="gel mass density (kg/m^3)")
parser.add_argument("--indentor-youngs", type=float, default=5.0e8,
                    help="indentor Young's modulus (Pa); high => effectively rigid")
parser.add_argument("--gel-bottom-bc", choices=["soft", "fixed"], default="soft",
                    help="bottom boundary: soft uses SoftPositionConstraint; fixed sets builtin.is_fixed on bottom vertices")
parser.add_argument("--gel-constraint-strength", type=float, default=100.0,
                    help="SoftPositionConstraint strength ratio for constrained gel bottom vertices")
parser.add_argument("--indentor-constraint-strength", type=float, default=100.0,
                    help="SoftPositionConstraint strength ratio for the kinematically driven indentor")
# --- stepping schedule ------------------------------------------------------
parser.add_argument("--press-steps", type=int, default=40, help="frames to lower the indentor")
parser.add_argument("--settle-steps", type=int, default=10, help="frames to settle after press")
parser.add_argument("--shear-steps", type=int, default=80, help="frames for the lateral drag")
parser.add_argument("--shear-settle", type=int, default=10, help="frames to settle after shear")
parser.add_argument("--dt", type=float, default=0.01, help="UIPC timestep (s)")
parser.add_argument("--save-trajectory", action="store_true",
                    help="store marker-field snapshots along the lateral loading path")
parser.add_argument("--traj-steps", type=int, default=8,
                    help="trajectory snapshots including normal state f=0 and final f=1")
parser.add_argument("--load-mode", choices=["linear", "ortho", "reverse"], default="linear",
                    help="lateral loading-path shape; same endpoint, different path")
# --- IPC convergence knobs --------------------------------------------------
parser.add_argument("--gel-res", type=int, default=12,
                    help="gel structured-mesh resolution (cells/footprint-axis); MESH refinement knob")
parser.add_argument("--gel-nz", type=int, default=0,
                    help="gel structured-mesh z layers; 0=auto (legacy round(lz/cell))")
parser.add_argument("--indentor-subdiv", type=int, default=2,
                    help="icosphere subdivisions for the (rigid, deterministic fan-meshed) indentor")
parser.add_argument("--eps-velocity", type=float, default=0.01,
                    help="friction smoothing velocity [m/s] (friction-model limit knob)")
parser.add_argument("--d-hat", type=float, default=1.0e-3, help="IPC barrier activation distance (m)")
parser.add_argument("--contact-resistance", type=float, default=1.0e9,
                    help="IPC contact resistance [Pa] (default_contact_resistance)")
parser.add_argument("--newton-max-iter", type=int, default=1024)
parser.add_argument("--velocity-tol", type=float, default=0.01, help="Newton convergence tol")
# --- convergence-sweep lists (comma-separated) ------------------------------
parser.add_argument("--gel-res-list", default="6,8,12,16,20",
                    help="comma-separated gel_res values (mesh refinement)")
parser.add_argument("--eps-velocity-list", default="0.02,0.01,0.005,0.002",
                    help="comma-separated eps_velocity values (friction limit)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--progress-file", default="/work/fem_progress_uipc.txt",
                    help="live progress log path (give shards distinct files to avoid clobber)")
parser.add_argument("--probe-contact-api", action="store_true",
                    help="log libuipc contact/force feature candidates to the progress file")
args = parser.parse_args()

import os

_PROG = args.progress_file


def flog(msg):
    try:
        with open(_PROG, "a") as f:
            f.write(str(msg) + "\n")
            f.flush()
    except OSError:
        pass


open(_PROG, "w").close()
flog("start: importing AppLauncher")

# --- make the IsaacLab + TacEx extension sources importable (same trick the
#     PhysX driver uses for /workspace/isaaclab/source/*). The tacex_uipc and
#     uipc bindings live under /workspace/tacex/source/*; without this the
#     `from tacex_uipc import ...` / `from uipc import ...` imports below fail.
import sys
import glob

for _root in ("/workspace/isaaclab/source", "/workspace/tacex/source"):
    for _p in glob.glob(_root + "/*"):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# --- launch Isaac headless (UipcSim needs a SimulationContext) --------------
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
flog("AppLauncher app created")

import time
import numpy as np
from scipy.spatial import cKDTree

from contact_imprint import mesh_contact_profile, mesh_object_surface

import isaaclab.sim as sim_utils

from tacex_uipc import UipcSim, UipcSimCfg

import uipc
import uipc.core
import uipc.geometry
from uipc import Animation, builtin, view
from uipc.constitution import ElasticModuli, SoftPositionConstraint, StableNeoHookean
from uipc.geometry import (
    GeometrySlot,
    SimplicialComplex,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh,
)

flog("imports OK")

# --- geometry / labelling, Z-up to match the PhysX driver -------------------
GEL = (args.gel_xy, args.gel_xy, args.gel_z)
GEL_TOP_Z = GEL[2]
MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]
G_STICK, G_PARTIAL, G_FULL = 0.04, 0.48, 1.0
GEOM_CODE = {"sphere": 0, "flat": 1, "cylinder": 2, "mesh": 3, "cuboid": 4, "ellipsoid": 5}
LOAD_MODES = ["linear", "ortho", "reverse"]


def label_mode(shear_mag, mu):
    """Cattaneo-Mindlin drive ratio thresholds (same as the PhysX driver)."""
    g = shear_mag / max(mu, 1e-6)
    if g < G_STICK:
        return 0
    if g < G_PARTIAL:
        return 1
    if g < G_FULL:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Surface-mesh generators (closed triangle meshes fed to wildmeshing)
# ---------------------------------------------------------------------------
def icosphere_surface(radius, subdiv=2, center=(0.0, 0.0, 0.0)):
    """Unit icosphere subdivided ``subdiv`` times, scaled to ``radius`` -> (V, F)."""
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
    for _ in range(subdiv):
        cache = {}
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(cache, a, b, vlist)
            bc = midpoint(cache, b, c, vlist)
            ca = midpoint(cache, c, a, vlist)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    V = np.array(vlist, dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    V = V + np.asarray(center, dtype=np.float64)
    return V, np.array(faces, dtype=np.uint32)


def cylinder_surface(radius, half_z, n_theta=48, n_z=4, center=(0.0, 0.0, 0.0)):
    """Closed Z-axis cylinder surface with deterministic vertex/face order."""
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    zvals = np.linspace(-half_z, half_z, int(n_z) + 1)
    verts = []
    for z in zvals:
        for th in theta:
            verts.append((cx + radius * np.cos(th), cy + radius * np.sin(th), cz + z))
    bottom_ci = len(verts)
    verts.append((cx, cy, cz - half_z))
    top_ci = len(verts)
    verts.append((cx, cy, cz + half_z))

    faces = []
    nt = int(n_theta)
    for iz in range(int(n_z)):
        row = iz * nt
        nxt = (iz + 1) * nt
        for i in range(nt):
            j = (i + 1) % nt
            faces.append((row + i, row + j, nxt + j))
            faces.append((row + i, nxt + j, nxt + i))
    for i in range(nt):
        j = (i + 1) % nt
        faces.append((bottom_ci, j, i))
        top_row = int(n_z) * nt
        faces.append((top_ci, top_row + i, top_row + j))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def cuboid_surface(half_x, half_y, half_z, center=(0.0, 0.0, 0.0)):
    """Closed box surface. Tier-2 input support is added separately."""
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    verts = np.array([
        [cx - half_x, cy - half_y, cz - half_z],
        [cx + half_x, cy - half_y, cz - half_z],
        [cx + half_x, cy + half_y, cz - half_z],
        [cx - half_x, cy + half_y, cz - half_z],
        [cx - half_x, cy - half_y, cz + half_z],
        [cx + half_x, cy - half_y, cz + half_z],
        [cx + half_x, cy + half_y, cz + half_z],
        [cx - half_x, cy + half_y, cz + half_z],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.uint32)
    return verts, faces


def ellipsoid_surface(rx, ry, rz, subdiv=2, center=(0.0, 0.0, 0.0)):
    """Icosphere scaled to an ellipsoid. Tier-3 input/schema support is separate."""
    V, F = icosphere_surface(1.0, subdiv=subdiv, center=(0.0, 0.0, 0.0))
    V = V * np.array([rx, ry, rz], dtype=np.float64) + np.asarray(center, dtype=np.float64)
    return V, F


def _bolt_hex_xy(radius, points_per_corner=4):
    """Rounded-hex footprint polygon for Tier-4 object #1."""
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
            # Quadratic corner arc with the true hex vertex as the control point.
            q = (1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * v + t ** 2 * b
            pts.append(q)
    return np.asarray(pts, dtype=np.float64)


def bolt_hex_surface(radius, half_z, center=(0.0, 0.0, 0.0)):
    """Closed rounded hex bolt-head prism, deterministic and convex."""
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


def fan_tet_sphere(V, F, center):
    """Deterministic tetrahedralisation of a CONVEX closed surface (V, F).

    Adds the sphere centre as one extra vertex and builds one tet per surface
    triangle (tri + centre). Used for the rigid indentor INSTEAD of wildmeshing:
    the indentor is kinematically driven and only its surface contacts the gel, so
    its interior tets are physically irrelevant — but wildmeshing is NON-deterministic
    (different mesh each run), which injects few-percent contact noise that contaminates
    the eps_velocity (friction) convergence axis. A fan mesh is identical every run.

    Valid because the sphere is convex: every (tri, centre) tet has positive volume
    and adjacent tets share exactly the internal face (edge, centre), so only the
    original triangles remain on the boundary. Returns (points Nx3, tets Mx4).
    """
    V = np.asarray(V, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    pts = np.vstack([V, c])
    ci = len(V)  # index of the centre vertex
    tets = np.array([[int(a), int(b), int(cc), ci] for a, b, cc in F], dtype=np.int64)
    # enforce positive signed volume (swap last two surface verts if negative)
    p = pts[tets]
    vol = np.einsum("ij,ij->i",
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0])
    neg = vol < 0
    tets[neg] = tets[neg][:, [1, 0, 2, 3]]
    return pts, tets


def structured_tet_box(size, gel_res, nz=None):
    """Deterministic structured tet mesh of the gel box -> (points Nx3, tets Mx4).

    A regular nx*ny*nz grid of hexes, each split into 6 tets (positive volume).
    Used INSTEAD of wildmeshing for the gel because a convergence study needs:
      * determinism  — same gel_res -> identical mesh (so the friction-axis sweep
        isolates eps_velocity instead of conflating it with random remeshing);
      * a FLAT regular top face — so top-face detection and marker sampling are
        well-posed (wildmeshing's irregular top collapsed the sampled field to a
        single vertex at fine resolution);
    gel_res = number of cells along each footprint axis (x=y).  When nz is
    None, z layers use the legacy round(lz/cell) rule, so they change in
    steps; pass nz explicitly to refine through the thickness.
    """
    lx, ly, lz = size
    nx = ny = int(gel_res)
    cell = lx / nx
    nz = max(1, int(round(lz / cell))) if nz is None else int(nz)
    xs = np.linspace(-lx / 2, lx / 2, nx + 1)
    ys = np.linspace(-ly / 2, ly / 2, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    # vertex grid, index = i + (nx+1)*(j + (ny+1)*k)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    # ravel Fortran-order so the flat index is i + (nx+1)*j + (nx+1)*(ny+1)*k,
    # matching vid() below (i fastest). C-order would put k fastest and break it.
    pts = np.stack([gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")],
                   axis=-1).astype(np.float64)

    def vid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    # standard 6-tet split sharing the cube main diagonal (corner 0 -> corner 7)
    splits = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
              (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # cube corners c0..c7 (x fastest, then y, then z)
                c = [vid(i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)) for m in range(8)]
                for a, b, cc, d in splits:
                    tets.append((c[a], c[b], c[cc], c[d]))
    tets = np.array(tets, dtype=np.int64)
    # enforce positive signed volume (swap last two verts if negative)
    p = pts[tets]
    vol = np.einsum("ij,ij->i",
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0])
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return pts, tets


def to_uipc_mesh(tet_points, tet_indices):
    """Build + label a libuipc tetmesh ready for contact."""
    mesh = tetmesh(tet_points.copy(), tet_indices.copy())
    label_surface(mesh)
    label_triangle_orient(mesh)
    mesh = flip_inward_triangles(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Marker sampling (identical to the PhysX driver)
# ---------------------------------------------------------------------------
def marker_grid(side):
    xs = np.linspace(-GEL[0] / 2 * 0.9, GEL[0] / 2 * 0.9, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def contact_profile(coords):
    """Stored mesh-raycast penetration profile for mesh-input Tier 4."""
    if args.indentor_geom != "mesh":
        return None
    params_row = np.array([
        0.0, 0.0, args.depth, args.indentor_r, 0.0, 0.0,
        args.mu, args.youngs, float(GEOM_CODE["mesh"]), indentor_r2(),
    ], dtype=np.float32)
    try:
        return mesh_contact_profile(
            coords,
            params_row,
            args.indentor_mesh or "bolt_hex",
            half_z=indentor_half_z(),
            r2=indentor_r2(),
            subdiv=args.indentor_subdiv,
        ).astype(np.float32)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def sample_to_markers(top_rest_xy, top_disp, coords):
    """Sample structured top-face displacements at marker coordinates.

    The gel mesh is a regular structured box, so the top face is a tensor-product
    grid. Bilinear interpolation avoids nearest-neighbor marker quantization when
    the top grid is coarser than the 32x32 marker layout. If a future mesh is not
    rectangular, fall back to the old nearest-vertex sampler.
    """
    xvals = np.unique(top_rest_xy[:, 0])
    yvals = np.unique(top_rest_xy[:, 1])
    nx, ny = len(xvals), len(yvals)
    if nx * ny != top_rest_xy.shape[0] or nx < 2 or ny < 2:
        tree = cKDTree(top_rest_xy)
        _, idx = tree.query(coords, k=1)
        return top_disp[idx], "nearest"

    grid = np.full((nx, ny, 3), np.nan, dtype=np.float64)
    ix = np.searchsorted(xvals, top_rest_xy[:, 0])
    iy = np.searchsorted(yvals, top_rest_xy[:, 1])
    grid[ix, iy] = top_disp
    if np.isnan(grid).any():
        tree = cKDTree(top_rest_xy)
        _, idx = tree.query(coords, k=1)
        return top_disp[idx], "nearest"

    cx = np.clip(coords[:, 0], xvals[0], xvals[-1])
    cy = np.clip(coords[:, 1], yvals[0], yvals[-1])
    i0 = np.clip(np.searchsorted(xvals, cx, side="right") - 1, 0, nx - 2)
    j0 = np.clip(np.searchsorted(yvals, cy, side="right") - 1, 0, ny - 2)
    tx = ((cx - xvals[i0]) / np.maximum(xvals[i0 + 1] - xvals[i0], 1e-12))[:, None]
    ty = ((cy - yvals[j0]) / np.maximum(yvals[j0 + 1] - yvals[j0], 1e-12))[:, None]

    f00 = grid[i0, j0]
    f10 = grid[i0 + 1, j0]
    f01 = grid[i0, j0 + 1]
    f11 = grid[i0 + 1, j0 + 1]
    field = ((1.0 - tx) * (1.0 - ty) * f00
             + tx * (1.0 - ty) * f10
             + (1.0 - tx) * ty * f01
             + tx * ty * f11)
    return field, "bilinear"


def contact_force_to_markers(uipc_sim, n_gel_verts, top, top_rest_xy, coords):
    """Read contact forces and sample top-surface traction on the gel.

    ``contact_gradient`` returns sparse global-vertex gradients for each contact
    primitive. The incremental-potential gradient contains ``dt**2``; therefore
    nodal force in N is ``-grad / dt**2``. Gel vertices are global indices
    ``[0, n_gel_verts)``. Before interpolation, each top-surface nodal force is
    divided by its structured-grid tributary area: interior ``dx*dy``, edge
    ``dx*dy/2``, corner ``dx*dy/4``. The two returned (M, 3) fields are therefore
    traction in N/m^2, split barrier/normal and friction. The two (3,) totals
    are exact nodal sums in N before sampling. All forces act on the gel, so a
    downward indentation has ``F_z < 0``.
    """
    feature = uipc_sim.world.features().find(uipc.core.ContactSystemFeature)
    channels = {
        "normal": np.zeros((n_gel_verts, 3), dtype=np.float64),
        "friction": np.zeros((n_gel_verts, 3), dtype=np.float64),
    }
    for prim_type in feature.contact_primitive_types():
        # Empty Geometry is the API's out-parameter; never pass the gel geometry.
        grad_geo = uipc.geometry.Geometry()
        feature.contact_gradient(prim_type, grad_geo)
        indices = np.asarray(view(grad_geo.instances().find("i")), dtype=np.int64).reshape(-1)
        grad = np.asarray(view(grad_geo.instances().find("grad")), dtype=np.float64).reshape(-1, 3)
        if len(indices) != len(grad):
            raise RuntimeError(
                f"contact_gradient {prim_type!s} has {len(indices)} indices for {len(grad)} gradients"
            )
        if str(prim_type).endswith("+N"):
            channel = "normal"
        elif str(prim_type).endswith("+F"):
            channel = "friction"
        else:
            raise RuntimeError(f"unknown contact primitive type: {prim_type!s}")
        gel_mask = (indices >= 0) & (indices < n_gel_verts)
        np.add.at(channels[channel], indices[gel_mask], -grad[gel_mask] / args.dt ** 2)

    xvals = np.unique(top_rest_xy[:, 0])
    yvals = np.unique(top_rest_xy[:, 1])
    nx, ny = len(xvals), len(yvals)
    if nx < 2 or ny < 2 or nx * ny != len(top):
        raise RuntimeError("contact traction requires a complete structured top-face grid")
    dx = (xvals[-1] - xvals[0]) / (nx - 1)
    dy = (yvals[-1] - yvals[0]) / (ny - 1)
    if dx <= 0.0 or dy <= 0.0:
        raise RuntimeError("contact traction requires positive structured-grid spacing")
    ix = np.searchsorted(xvals, top_rest_xy[:, 0])
    iy = np.searchsorted(yvals, top_rest_xy[:, 1])
    areas = np.full(len(top), dx * dy, dtype=np.float64)
    areas[(ix == 0) | (ix == nx - 1)] *= 0.5
    areas[(iy == 0) | (iy == ny - 1)] *= 0.5

    normal_total = channels["normal"].sum(axis=0)
    friction_total = channels["friction"].sum(axis=0)
    normal, marker_sampling = sample_to_markers(
        top_rest_xy, channels["normal"][top] / areas[:, None], coords
    )
    friction, friction_sampling = sample_to_markers(
        top_rest_xy, channels["friction"][top] / areas[:, None], coords
    )
    if marker_sampling != friction_sampling:
        raise RuntimeError("normal and friction force marker sampling disagree")
    return (
        normal.astype(np.float32),
        friction.astype(np.float32),
        normal_total.astype(np.float32),
        friction_total.astype(np.float32),
        marker_sampling,
    )


# ---------------------------------------------------------------------------
# Indentor motion schedule (press -> settle -> shear -> settle), rigid offset
# ---------------------------------------------------------------------------
def build_schedule():
    """Per-frame rigid translation (dx, dy, dz) applied to the indentor.

    The indentor starts with its lowest point just touching the gel top
    (gap = d_hat). Press lowers the centre by (depth + gap) so the lowest point
    penetrates ``depth``; shear then translates the centre by the requested
    lateral vector.
    Frame indexing is 1-based (libuipc advances before the first animator call).
    """
    n_press, n_settle = args.press_steps, args.settle_steps
    n_shear, n_shear_settle = args.shear_steps, args.shear_settle
    sx, sy = shear_xy()
    gap = args.d_hat
    z_down = args.depth + gap  # total downward centre travel
    sched = []
    for k in range(n_press):
        f = (k + 1) / n_press
        sched.append((0.0, 0.0, -z_down * f))
    for _ in range(n_settle):
        sched.append((0.0, 0.0, -z_down))
    for k in range(n_shear):
        f = (k + 1) / n_shear
        px, py = path_xy(f, sx, sy, args.load_mode)
        sched.append((px, py, -z_down))
    for _ in range(n_shear_settle):
        sched.append((sx, sy, -z_down))
    return sched, gap


def path_xy(f, sx, sy, mode):
    """Endpoint-preserving loading paths, matching the legacy PhysX trajectory schema."""
    if mode == "ortho":
        if f <= 0.5:
            return sx * (f / 0.5), 0.0
        return sx, sy * ((f - 0.5) / 0.5)
    if mode == "reverse":
        s = 1.5 * (f / 0.66) if f <= 0.66 else 1.5 - 0.5 * ((f - 0.66) / 0.34)
        return sx * s, sy * s
    return sx * f, sy * f


def shear_xy():
    """Return the lateral endpoint vector.

    ``--shear`` remains the original +x shorthand. Paired PhysX comparisons use
    ``--shear-x/--shear-y`` so the IPC load matches one sampled PhysX frame.
    """
    if args.shear_x is None:
        return float(args.shear), 0.0
    return float(args.shear_x), float(0.0 if args.shear_y is None else args.shear_y)


def indentor_half_z():
    return float(args.indentor_r if args.indentor_half_z <= 0 else args.indentor_half_z)


def indentor_r2():
    return float(args.indentor_r if args.indentor_r2 <= 0 else args.indentor_r2)


def make_indentor_tetmesh(center):
    """Return (tet_points, tet_indices, bottom_off) for the requested indentor."""
    geom = args.indentor_geom
    hz = indentor_half_z()
    if geom == "sphere":
        bottom_off = float(args.indentor_r)
        V, F = icosphere_surface(args.indentor_r, subdiv=args.indentor_subdiv, center=center)
    elif geom == "cylinder":
        bottom_off = hz
        V, F = cylinder_surface(args.indentor_r, hz, center=center)
    elif geom == "cuboid":
        bottom_off = hz
        V, F = cuboid_surface(args.indentor_r, args.indentor_r, hz, center=center)
    elif geom == "ellipsoid":
        bottom_off = hz
        V, F = ellipsoid_surface(args.indentor_r, indentor_r2(), hz,
                                 subdiv=args.indentor_subdiv, center=center)
    elif geom == "mesh":
        try:
            V0, F, bottom_off = mesh_object_surface(
                args.indentor_mesh or "bolt_hex",
                args.indentor_r,
                hz,
                r2=indentor_r2(),
                subdiv=args.indentor_subdiv,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        V = V0 + np.asarray(center, dtype=np.float64)
    else:
        raise SystemExit(f"unknown indentor geom {geom!r}")
    return (*fan_tet_sphere(V, F, center), bottom_off)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def build_scene(uipc_sim, gel_res):
    """Create gel pad + sphere indentor in the UIPC scene.

    Returns a dict of handles used during stepping and readback.
    """
    scene = uipc_sim.scene
    snh = StableNeoHookean()
    spc = SoftPositionConstraint()

    # contact: rely on the global default friction ratio (= mu) set in the cfg.
    contact_tabular = scene.contact_tabular()
    default_element = contact_tabular.default_element()

    # --- gel pad (deformable) ------------------------------------------------
    # Structured tet box (NOT wildmeshing): deterministic + flat regular top so
    # the convergence sweep and marker sampling are well-posed (see structured_tet_box).
    flog(f"  meshing gel (structured, gel_res={gel_res}) ...")
    gel_pts, gel_tets = structured_tet_box(
        GEL, gel_res, args.gel_nz if args.gel_nz > 0 else None
    )
    flog(f"  gel tets: {gel_pts.shape[0]} verts, {gel_tets.shape[0]} tets")
    gel_mesh = to_uipc_mesh(gel_pts, gel_tets)
    # youngs_poisson takes Young's modulus in SI Pa (TacEx passes ``youngs*MPa``,
    # i.e. an already-SI value); args.youngs is already in Pa, so pass it raw.
    moduli = ElasticModuli.youngs_poisson(args.youngs, args.poisson)
    snh.apply_to(gel_mesh, moduli, mass_density=args.gel_density)
    default_element.apply_to(gel_mesh)
    # Dirichlet BC: pin the bottom face (z = 0). The historical path used a
    # SoftPositionConstraint, while the fixed path sets UIPC's hard is_fixed flag.
    spc.apply_to(gel_mesh, args.gel_constraint_strength)
    gel_pos0 = gel_mesh.positions().view().reshape(-1, 3).copy()
    # structured mesh => bottom verts sit exactly at z=0; a tight tol is safe.
    gel_bottom_mask = gel_pos0[:, 2] <= (gel_pos0[:, 2].min() + 1e-6)
    if args.gel_bottom_bc == "fixed":
        is_fixed = gel_mesh.vertices().find(builtin.is_fixed)
        if not is_fixed:
            is_fixed = gel_mesh.vertices().create(builtin.is_fixed, 0)
        view(is_fixed)[gel_bottom_mask] = 1
    gel_object = scene.objects().create("gel")
    gel_slot, _ = gel_object.geometries().create(gel_mesh)

    # --- indentor (stiff SNH, all vertices prescribed => kinematic) ----------
    # Its interior tets are physically irrelevant; deterministic fan tets keep
    # contact-surface noise out of geometry-OOD comparisons.
    bottom_off0 = args.indentor_r if args.indentor_geom == "sphere" else indentor_half_z()
    c0 = (0.0, 0.0, GEL_TOP_Z + bottom_off0 + args.d_hat)
    flog(f"  meshing indentor {args.indentor_geom} (deterministic fan) ...")
    ind_pts, ind_tets, bottom_off = make_indentor_tetmesh(c0)
    if abs(bottom_off - bottom_off0) > 1e-12:
        raise RuntimeError("indentor bottom offset changed after mesh creation")
    flog(f"  indentor tets: {ind_pts.shape[0]} verts, {ind_tets.shape[0]} tets")
    ind_mesh = to_uipc_mesh(ind_pts, ind_tets)
    ind_moduli = ElasticModuli.youngs_poisson(args.indentor_youngs, 0.45)
    snh.apply_to(ind_mesh, ind_moduli, mass_density=args.gel_density)
    default_element.apply_to(ind_mesh)
    spc.apply_to(ind_mesh, args.indentor_constraint_strength)
    ind_pos0 = ind_mesh.positions().view().reshape(-1, 3).copy()
    ind_object = scene.objects().create("indentor")
    ind_slot, _ = ind_object.geometries().create(ind_mesh)

    return {
        "gel_object": gel_object,
        "gel_slot": gel_slot,
        "gel_pos0": gel_pos0,
        "gel_bottom_mask": gel_bottom_mask,
        "ind_object": ind_object,
        "ind_slot": ind_slot,
        "ind_pos0": ind_pos0,
    }


def install_animators(uipc_sim, handles, schedule):
    """Register per-object animators: pin gel bottom, drive indentor along schedule."""
    scene = uipc_sim.scene
    animator = scene.animator()
    gel_pos0 = handles["gel_pos0"]
    gel_bottom_mask = handles["gel_bottom_mask"]
    ind_pos0 = handles["ind_pos0"]
    n_sched = len(schedule)

    def assign_positions(aim, mask, src):
        """Write Nx3 ``src`` into ``aim[mask]`` for either Nx3 or Nx3x1 UIPC views.

        IMPORTANT: assign with a *single fancy-index* directly on ``aim`` so the
        write propagates. ``aim[mask][:] = ...`` would write into a throw-away
        copy (numpy boolean-mask getitem copies) and silently lose the update.
        """
        src = np.asarray(src, dtype=np.float64)
        if aim.ndim == 3 and aim.shape[-1] == 1:
            aim[mask, :, 0] = src
        else:
            aim[mask] = src

    def animate_gel(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim = view(geo.vertices().find(builtin.aim_position))
        is_c[gel_bottom_mask] = 1
        assign_positions(aim, gel_bottom_mask, gel_pos0[gel_bottom_mask])

    def animate_indentor(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim = view(geo.vertices().find(builtin.aim_position))
        frame = int(info.frame())
        idx = min(max(frame - 1, 0), n_sched - 1)
        dx, dy, dz = schedule[idx]
        is_c[:] = 1
        assign_positions(aim, slice(None), ind_pos0 + np.array([dx, dy, dz], dtype=np.float64))

    animator.insert(handles["gel_object"], animate_gel)
    animator.insert(handles["ind_object"], animate_indentor)


# ---------------------------------------------------------------------------
# One run: build sim, step through indent+shear, read the gel field back
# ---------------------------------------------------------------------------
def make_uipc_cfg(eps_velocity):
    return UipcSimCfg(
        dt=args.dt,
        gravity=(0.0, 0.0, -9.8),
        ground_normal=(0.0, 0.0, 1.0),
        # Keep UIPC's default half-plane well below the gel. The gel bottom is
        # pinned at z=0; placing the ground at z=0 makes the world invalid.
        ground_height=-args.gel_z,
        logger_level="Error",
        newton=UipcSimCfg.Newton(max_iter=args.newton_max_iter, velocity_tol=args.velocity_tol),
        contact=UipcSimCfg.Contact(
            enable=True,
            enable_friction=True,
            default_friction_ratio=args.mu,
            default_contact_resistance=args.contact_resistance,
            d_hat=args.d_hat,
            eps_velocity=eps_velocity,
        ),
    )


def probe_contact_api(uipc_sim, geo, phase):
    """Record the contact-force API surface exposed by this libuipc build.

    This is deliberately reflection-only: it records ``dir()`` and ``hasattr()``
    results for the loaded modules, solver, geometry, and vertices.  It never
    synthesizes force from displacement or solver state, nor treats any generic
    dynamic/internal force as contact GT.  TacEx/libuipc bindings are image-
    specific, so this diagnostic must identify an advertised contact/friction
    feature before a data-query path is added.
    """
    if not args.probe_contact_api:
        return
    flog(f"CONTACT_API probe phase: {phase}")
    keywords = ("force", "contact", "friction", "traction", "impulse")
    import importlib
    import pkgutil
    import re
    import tacex_uipc
    import uipc
    import uipc.geometry as uipc_geometry

    if phase == "after_setup":
        flog(f"CONTACT_API module files: uipc={getattr(uipc, '__file__', '?')} "
             f"core={getattr(uipc.core, '__file__', '?')}")
        source_hits = 0
        for root in ("/workspace/tacex/source", "/workspace/uipc/source"):
            if not os.path.isdir(root):
                continue
            for parent, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in (".git", "build", "_build")]
                for filename in files:
                    if not filename.endswith((".h", ".hpp", ".cpp", ".cu", ".py")):
                        continue
                    path = os.path.join(parent, filename)
                    try:
                        if os.path.getsize(path) > 2_000_000:
                            continue
                        with open(path, errors="ignore") as f:
                            for line_no, line in enumerate(f, start=1):
                                if "contact_gradient" in line:
                                    flog(f"CONTACT_API source {path}:{line_no}: {line.strip()}")
                                    source_hits += 1
                                    if source_hits >= 32:
                                        break
                    except OSError:
                        continue
                    if source_hits >= 32:
                        break
                if source_hits >= 32:
                    break
        flog(f"CONTACT_API source contact_gradient hits: {source_hits}")
        source_root = "/workspace/tacex/source/tacex_uipc/libuipc/src/backends/cuda/contact_system"
        for filename, first, last in (
            ("contact_exporter.cu", 20, 48),
            ("contact_exporter_manager.cu", 50, 72),
            ("simplex_normal_contact.cu", 338, 370),
            ("simplex_frictional_contact.cu", 342, 374),
        ):
            path = os.path.join(source_root, filename)
            try:
                with open(path) as f:
                    lines = f.readlines()
                for line_no in range(first, min(last, len(lines)) + 1):
                    flog(f"CONTACT_API sourcectx {filename}:{line_no}: "
                         f"{lines[line_no - 1].rstrip()}")
            except OSError as exc:
                flog(f"CONTACT_API sourcectx {filename}: unavailable "
                     f"({type(exc).__name__}: {exc})")
        for filename in ("simplex_normal_contact.cu", "simplex_frictional_contact.cu"):
            path = os.path.join(source_root, filename)
            try:
                with open(path) as f:
                    lines = f.readlines()
                for line_no, line in enumerate(lines, start=1):
                    if "PT_gradients" not in line:
                        continue
                    lo, hi = max(1, line_no - 3), min(len(lines), line_no + 3)
                    for context_no in range(lo, hi + 1):
                        flog(f"CONTACT_API gradientsctx {filename}:{context_no}: "
                             f"{lines[context_no - 1].rstrip()}")
            except OSError as exc:
                flog(f"CONTACT_API gradientsctx {filename}: unavailable "
                     f"({type(exc).__name__}: {exc})")
        for filename in (
            "contact_system.cu", "simplex_normal_contact.cu",
            "simplex_frictional_contact.cu",
        ):
            path = os.path.join(source_root, filename)
            try:
                with open(path) as f:
                    lines = f.readlines()
                hits = 0
                for line_no, line in enumerate(lines, start=1):
                    if not re.search(r"\\b(dt|time_step|time\\(\\))\\b", line):
                        continue
                    flog(f"CONTACT_API timectx {filename}:{line_no}: {line.rstrip()}")
                    hits += 1
                    if hits >= 32:
                        break
                flog(f"CONTACT_API timectx {filename}: hits={hits}")
            except OSError as exc:
                flog(f"CONTACT_API timectx {filename}: unavailable "
                     f"({type(exc).__name__}: {exc})")

    def log_surface(label, obj):
        try:
            names = sorted(name for name in dir(obj) if not name.startswith("__"))
        except Exception as exc:
            flog(f"CONTACT_API {label}: dir unavailable ({type(exc).__name__}: {exc})")
            return
        candidates = [name for name in names
                      if any(word in name.lower() for word in keywords)]
        flog(f"CONTACT_API {label} all: " + ",".join(names))
        flog(f"CONTACT_API {label} keyword candidates: " + ",".join(candidates))
        for name in candidates:
            try:
                exists = hasattr(obj, name)
                value = getattr(obj, name) if exists else None
                flog(f"CONTACT_API {label}.{name}: hasattr={exists} "
                     f"type={type(value).__name__ if exists else 'none'} "
                     f"callable={callable(value) if exists else False}")
            except Exception as exc:
                flog(f"CONTACT_API {label}.{name}: inspect failed "
                     f"({type(exc).__name__}: {exc})")

    modules = sorted(m.name for m in pkgutil.iter_modules(uipc.__path__))
    flog("CONTACT_API uipc child modules: " + ",".join(modules))
    extra_modules = {}
    for module_name in ("backend", "core", "diff_sim", "stats", "constitution"):
        try:
            extra_modules[module_name] = importlib.import_module(f"uipc.{module_name}")
            flog(f"CONTACT_API imported uipc.{module_name}: yes")
        except Exception as exc:
            flog(f"CONTACT_API imported uipc.{module_name}: no "
                 f"({type(exc).__name__}: {exc})")
    vertices = geo.vertices()
    surfaces = [
        ("uipc", uipc),
        ("tacex_uipc", tacex_uipc),
        ("uipc.builtin", builtin),
        ("uipc.geometry", uipc_geometry),
        ("uipc.Engine", uipc.Engine),
        ("uipc.Scene", uipc.Scene),
        ("uipc.World", uipc.World),
        ("UipcSim", UipcSim),
        ("UipcSim.instance", uipc_sim),
        ("UipcSimCfg", UipcSimCfg),
        ("uipc_sim.engine", uipc_sim.engine),
        ("uipc_sim.scene", uipc_sim.scene),
        ("uipc_sim.world", uipc_sim.world),
        ("SimplicialComplex", SimplicialComplex),
        ("gel_geo", geo),
        ("gel_vertices", vertices),
    ]
    surfaces.extend((f"uipc.{name}", module) for name, module in extra_modules.items())
    core = extra_modules.get("core")
    if core is not None:
        for class_name in (
            "ContactElement", "ContactModel", "ContactModelCollection",
            "ContactSystemFeature", "ContactTabular",
        ):
            if hasattr(core, class_name):
                surfaces.append((f"uipc.core.{class_name}", getattr(core, class_name)))
    for label, obj in surfaces:
        log_surface(label, obj)

    def log_call_surface(label, method):
        try:
            value = method()
            flog(f"CONTACT_API {label}(): returned type={type(value).__name__}")
            log_surface(f"{label}()", value)
            return value
        except Exception as exc:
            flog(f"CONTACT_API {label}(): unavailable ({type(exc).__name__}: {exc})")
            return None

    def log_doc(label, value):
        try:
            doc = getattr(value, "__doc__", None)
            text = "" if doc is None else " ".join(str(doc).split())
            flog(f"CONTACT_API {label} doc: {text[:1200]}")
        except Exception as exc:
            flog(f"CONTACT_API {label} doc unavailable ({type(exc).__name__}: {exc})")

    # These are the only advertised routes from the live solver toward its
    # contact models/features.  Calling them here exposes their API surface but
    # does not request a force value or reinterpret a displacement as one.
    call_results = {}
    for label, method in (
        ("uipc_sim.scene.contact_tabular", uipc_sim.scene.contact_tabular),
        ("uipc_sim.scene.diff_sim", uipc_sim.scene.diff_sim),
        ("uipc_sim.engine.features", uipc_sim.engine.features),
        ("uipc_sim.world.features", uipc_sim.world.features),
    ):
        call_results[label] = log_call_surface(label, method)

    contact_table = call_results["uipc_sim.scene.contact_tabular"]
    if contact_table is not None:
        for name in ("contact_models", "default_model", "default_element", "element_count"):
            if hasattr(contact_table, name):
                log_call_surface(f"uipc_sim.scene.contact_tabular().{name}",
                                 getattr(contact_table, name))

    for label in ("uipc_sim.engine.features", "uipc_sim.world.features"):
        collection = call_results[label]
        if collection is not None and hasattr(collection, "to_json"):
            try:
                flog(f"CONTACT_API {label}().to_json: {collection.to_json()}")
            except Exception as exc:
                flog(f"CONTACT_API {label}().to_json: unavailable "
                     f"({type(exc).__name__}: {exc})")
        if collection is not None and hasattr(collection, "find"):
            log_doc(f"{label}().find", collection.find)

    if core is not None and hasattr(core, "ContactSystemFeature"):
        contact_feature_type = core.ContactSystemFeature
        for name in (
            "FeatureName", "contact_energy", "contact_gradient",
            "contact_hessian", "contact_primitive_types",
        ):
            if hasattr(contact_feature_type, name):
                log_doc(f"uipc.core.ContactSystemFeature.{name}",
                        getattr(contact_feature_type, name))
        if hasattr(contact_feature_type, "FeatureName"):
            flog("CONTACT_API uipc.core.ContactSystemFeature.FeatureName: "
                 f"{contact_feature_type.FeatureName}")
        for label in ("uipc_sim.engine.features", "uipc_sim.world.features"):
            collection = call_results[label]
            if collection is None:
                continue
            try:
                feature = collection.find(contact_feature_type)
                flog(f"CONTACT_API {label}().find(ContactSystemFeature): "
                     f"returned type={type(feature).__name__}")
                log_surface(f"{label}().find(ContactSystemFeature)", feature)
                if hasattr(feature, "contact_primitive_types"):
                    flog(f"CONTACT_API {label}().find(ContactSystemFeature)"
                         f".contact_primitive_types(): {feature.contact_primitive_types()}")
            except Exception as exc:
                flog(f"CONTACT_API {label}().find(ContactSystemFeature): unavailable "
                     f"({type(exc).__name__}: {exc})")

    # ``contact_element_id`` is an advertised contact label in this build.  Check
    # whether it is actually materialized on gel vertices, but do not infer a
    # force from labels if it is absent or lacks a force counterpart.
    if hasattr(builtin, "contact_element_id"):
        try:
            attr = vertices.find(builtin.contact_element_id)
            if attr:
                arr = np.asarray(view(attr))
                flog("CONTACT_API builtin.contact_element_id feature: "
                     f"shape={arr.shape} dtype={arr.dtype}")
            else:
                flog("CONTACT_API builtin.contact_element_id feature: not materialized")
        except Exception as exc:
            flog("CONTACT_API builtin.contact_element_id feature: unavailable "
                 f"({type(exc).__name__}: {exc})")

    try:
        objects = uipc_sim.uipc_objects
        if hasattr(objects, "items"):
            flog("CONTACT_API uipc_objects keys: " + ",".join(str(k) for k in objects))
            for name, obj in objects.items():
                log_surface(f"uipc_objects[{name!r}]", obj)
        else:
            log_surface("uipc_objects", objects)
            for index, obj in enumerate(objects):
                if index >= 16:
                    flog("CONTACT_API uipc_objects: truncated after 16 entries")
                    break
                log_surface(f"uipc_objects[{index}]", obj)
    except Exception as exc:
        flog(f"CONTACT_API uipc_objects: inspect failed ({type(exc).__name__}: {exc})")


def probe_contact_gradient(uipc_sim, geo, top):
    """Exercise the verified solver gradient readback without writing GT data.

    ``ContactSystemFeature.contact_gradient`` is the feature API exposed by this
    image.  A zeroed clone is used as its documented ``vert_grad`` output buffer,
    so this diagnostic cannot mutate the live simulation geometry.  The result is
    only logged here; production force serialization is deliberately deferred
    until its units/sign and top-surface support are validated.
    """
    if not args.probe_contact_api:
        return
    try:
        from uipc.core import ContactSystemFeature
        feature = uipc_sim.world.features().find(ContactSystemFeature)
        primitive_types = feature.contact_primitive_types()
        offsets = None
        for name in ("_system_vertex_offsets", "_surf_vertex_offsets"):
            try:
                value = getattr(uipc_sim, name)
                flog(f"CONTACT_API uipc_sim.{name}: type={type(value).__name__} value={value!r}")
                if name == "_system_vertex_offsets":
                    offsets = value
            except Exception as exc:
                flog(f"CONTACT_API uipc_sim.{name}: unavailable "
                     f"({type(exc).__name__}: {exc})")
        n_gel = geo.positions().view().reshape(-1, 3).shape[0]
        try:
            gel_offset = int(np.asarray(offsets).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            gel_offset = 0
        gel_grad = np.zeros((n_gel, 3), dtype=np.float64)
        for builtin_name in ("global_vertex_offset", "dof_offset"):
            try:
                attr = geo.vertices().find(getattr(builtin, builtin_name))
                if attr:
                    values = np.asarray(view(attr))
                    flog(f"CONTACT_API gel {builtin_name}: shape={values.shape} "
                         f"dtype={values.dtype} sample={values.reshape(-1)[:8].tolist()}")
                else:
                    flog(f"CONTACT_API gel {builtin_name}: not materialized")
            except Exception as exc:
                flog(f"CONTACT_API gel {builtin_name}: unavailable "
                     f"({type(exc).__name__}: {exc})")
        for prim_type in primitive_types:
            grad_geo = geo.clone()
            feature.contact_gradient(prim_type, grad_geo)
            instances = grad_geo.instances()
            i_attr = instances.find("i")
            grad_attr = instances.find("grad")
            if not i_attr or not grad_attr:
                flog(f"CONTACT_API gradient {prim_type}: no i/grad instances")
                continue
            indices = np.asarray(view(i_attr), dtype=np.int64).reshape(-1)
            grad = np.asarray(view(grad_attr), dtype=np.float64).reshape(-1, 3)
            gel_mask = (indices >= gel_offset) & (indices < gel_offset + n_gel)
            if np.any(gel_mask):
                np.add.at(gel_grad, indices[gel_mask] - gel_offset, grad[gel_mask])
            flog(f"CONTACT_API gradient {prim_type}: n={len(indices)} "
                 f"index_minmax=({int(indices.min()) if len(indices) else 'na'},"
                 f"{int(indices.max()) if len(indices) else 'na'}) "
                 f"finite={bool(np.isfinite(grad).all())} l2={np.linalg.norm(grad):.9g} "
                 f"sum={grad.sum(axis=0).tolist()} "
                 f"sample_i={indices[:8].tolist()} sample_grad={grad[:3].tolist()}")
        raw_sum = gel_grad.sum(axis=0)
        raw_top_sum = gel_grad[top].sum(axis=0)
        force_candidate = -gel_grad / max(args.dt ** 2, 1e-12)
        hertz = ((4.0 / 3.0) * (args.youngs / (1.0 - args.poisson ** 2))
                 * np.sqrt(args.indentor_r) * args.depth ** 1.5)
        flog("CONTACT_API gel gradient aggregate: "
             f"gel_offset={gel_offset} raw_sum={raw_sum.tolist()} "
             f"raw_top_sum={raw_top_sum.tolist()} "
             f"force_candidate_sum={force_candidate.sum(axis=0).tolist()} "
             f"force_candidate_top_sum={force_candidate[top].sum(axis=0).tolist()} "
             f"hertz_halfspace_N={hertz:.9g}")
    except Exception as exc:
        flog(f"CONTACT_API gradient probe: unavailable ({type(exc).__name__}: {exc})")


def _trajectory_snap_indices():
    if not args.save_trajectory:
        return []
    if args.traj_steps < 2:
        raise SystemExit("--traj-steps must be >=2 when --save-trajectory is set")
    n_press, n_settle = args.press_steps, args.settle_steps
    n_shear = args.shear_steps
    shear_start = n_press + n_settle
    out = []
    for frac in np.linspace(0.0, 1.0, args.traj_steps):
        if frac <= 1e-9:
            idx = shear_start - 1
        else:
            idx = shear_start + int(np.ceil(frac * n_shear)) - 1
        out.append(max(0, idx))
    return out


def run_one(sim, gel_res, eps_velocity, coords, verbose=False):
    """Run a single (gel_res, eps_velocity) indent+shear; return field + scalars."""
    run_t0 = time.perf_counter()
    schedule, _gap = build_schedule()
    uipc_cfg = make_uipc_cfg(eps_velocity)
    uipc_sim = UipcSim(uipc_cfg)

    handles = build_scene(uipc_sim, gel_res)
    install_animators(uipc_sim, handles, schedule)

    # BATCH-SAFETY: setup_sim() registers a physics callback named "uicp_step" on
    # the shared SimulationContext. On the 2nd+ run in one process (batch mode) a
    # stale callback from the previous (now-deleted) UipcSim would collide -> remove
    # it first. (run_one steps via uipc_sim.step() directly, so the callback is not
    # functionally used here, but the duplicate-name registration would still raise.)
    try:
        if sim.physics_callback_exists("uicp_step"):
            sim.remove_physics_callback("uicp_step")
    except Exception:
        pass

    uipc_sim.setup_sim()

    t0 = time.perf_counter()
    n_total = len(schedule)
    traj_indices = _trajectory_snap_indices()
    traj_fields = []
    traj_cursor = 0
    for step in range(n_total):
        uipc_sim.step()
        if traj_cursor < len(traj_indices) and step >= traj_indices[traj_cursor]:
            cur = handles["gel_slot"].geometry().positions().view().reshape(-1, 3)
            rest = handles["gel_pos0"]
            disp_step = cur - rest
            top = np.where(rest[:, 2] >= rest[:, 2].max() - 1e-6)[0]
            marker_step, _ = sample_to_markers(rest[top, :2], disp_step[top], coords)
            traj_fields.append(marker_step.astype(np.float32))
            traj_cursor += 1
        if verbose and (step % 20 == 0 or step == n_total - 1):
            cur = handles["gel_slot"].geometry().positions().view().reshape(-1, 3)
            d = cur - handles["gel_pos0"]
            flog(f"    step {step+1}/{n_total} peak_uz={d[:,2].min():.5f} "
                 f"max|tang|={np.linalg.norm(d[:,:2],axis=1).max():.5f}")
    solve_time = time.perf_counter() - t0

    # readback: current gel vertex positions -> displacement field
    cur = handles["gel_slot"].geometry().positions().view().reshape(-1, 3).copy()
    rest = handles["gel_pos0"]
    disp = cur - rest

    # top-face vertices (structured mesh => exactly at z=GEL_TOP_Z at rest)
    top = np.where(rest[:, 2] >= rest[:, 2].max() - 1e-6)[0]
    top_markers, marker_sampling = sample_to_markers(rest[top, :2], disp[top], coords)  # (M, 3)
    (contact_force_normal, contact_force_friction,
     contact_force_total_normal, contact_force_total_friction,
     force_marker_sampling) = contact_force_to_markers(
         uipc_sim, rest.shape[0], top, rest[top, :2], coords
     )

    tnorm = np.linalg.norm(top_markers[:, :2], axis=1)
    scalars = {
        "gel_res": int(gel_res),
        "gel_nz": (int(args.gel_nz) if args.gel_nz > 0 else
                   max(1, int(round(GEL[2] / (GEL[0] / gel_res))))),
        "eps_velocity": float(eps_velocity),
        "n_tet_verts": int(rest.shape[0]),
        "n_top_verts": int(top.shape[0]),
        "peak_uz": float(top_markers[:, 2].min()),
        "max_tang": float(tnorm.max()),
        "mean_tang": float(tnorm.mean()),
        "net_tang_x": float(top_markers[:, 0].mean()),
        "net_tang_y": float(top_markers[:, 1].mean()),
        "solve_time_s": float(solve_time),
        "marker_sampling": marker_sampling,
        "force_marker_sampling": force_marker_sampling,
        "contact_force_total_normal": contact_force_total_normal,
        "contact_force_total_friction": contact_force_total_friction,
        "load_mode": args.load_mode,
    }
    # free the engine before the next setting (fresh UipcSim per run). In batch
    # mode (many runs/process) also drop the stale callback + force GC so libuipc
    # GPU buffers are released and don't accumulate toward OOM.
    try:
        if sim.physics_callback_exists("uicp_step"):
            sim.remove_physics_callback("uicp_step")
    except Exception:
        pass
    del uipc_sim
    import gc
    gc.collect()
    # Full run_one wall time: scene/config construction, setup, stepping,
    # readback, marker/force sampling, teardown and GC. Save/IO and the one-time
    # Isaac/container boot remain outside and are timed by the shell wrapper.
    scalars["run_time_s"] = float(time.perf_counter() - run_t0)
    return top_markers.astype(np.float32), contact_force_normal, contact_force_friction, scalars, (
        np.stack(traj_fields).astype(np.float32) if traj_fields else None
    )


# ---------------------------------------------------------------------------
# Field-distance helper for the convergence report
# ---------------------------------------------------------------------------
def rel_l2(a, b):
    denom = np.linalg.norm(b) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def channel_rel_l2(field_a, field_b):
    """rel-L2 split into normal (uz) and tangential (uxy), like the paired study."""
    n = rel_l2(field_a[:, 2], field_b[:, 2])
    t = rel_l2(field_a[:, :2], field_b[:, :2])
    o = rel_l2(field_a, field_b)
    return {"overall": o, "normal": n, "tangential": t}


# ---------------------------------------------------------------------------
# Save (PhysX-compatible npz)
# ---------------------------------------------------------------------------
def save_field(out_dir, coords, field, contact_force_normal, contact_force_friction, scalars, traj=None):
    os.makedirs(out_dir, exist_ok=True)
    sx, sy = shear_xy()
    # params row layout extends isaac_extract_shear.py:
    # [cx, cy, depth, R, sx, sy, mu, youngs, geom_code, R2]
    # R2 defaults to R for backward-compatible geometries and is the y semi-axis
    # for ellipsoid Tier-3 data.
    params = np.array([[0.0, 0.0, args.depth, args.indentor_r, sx, sy,
                        args.mu, args.youngs, float(GEOM_CODE[args.indentor_geom]),
                        indentor_r2()]], dtype=np.float32)
    # mode is the Cattaneo-Mindlin slip class. The PhysX sweep labels it from the
    # SAMPLED drive ratio g. For ad-hoc --single runs the shear is just a fixed
    # lateral TRAVEL in metres, so a faithful g is unavailable -> store -1. Scripted
    # production sweeps pass --drive-ratio explicitly, so their labels match PhysX.
    mode_value = -1 if args.drive_ratio is None else label_mode(float(args.drive_ratio) * args.mu, args.mu)
    mode = np.array([mode_value], dtype=np.int32)
    payload = dict(
        params=params,
        coords=coords.astype(np.float32),
        disp=field[None, ...].astype(np.float32),  # (1, M, 3)
        # N/m^2 traction acting on gel; downward indentation has negative F_z.
        contact_force_normal=contact_force_normal[None, ...].astype(np.float32),
        contact_force_friction=contact_force_friction[None, ...].astype(np.float32),
        # Exact N sums over gel vertices before any marker interpolation.
        contact_force_total_normal=np.asarray(
            scalars["contact_force_total_normal"], dtype=np.float32
        ).reshape(1, 3),
        contact_force_total_friction=np.asarray(
            scalars["contact_force_total_friction"], dtype=np.float32
        ).reshape(1, 3),
        mode=mode,
        solve_time_s=np.array([scalars["solve_time_s"]], dtype=np.float32),
        run_time_s=np.array([scalars["run_time_s"]], dtype=np.float32),
        gel_res=np.array([scalars["gel_res"]], dtype=np.int32),
        gel_nz=np.array([scalars["gel_nz"]], dtype=np.int32),
        gel_xy=np.array([args.gel_xy], dtype=np.float32),
        gel_z=np.array([args.gel_z], dtype=np.float32),
        eps_velocity=np.array([scalars["eps_velocity"]], dtype=np.float32),
        velocity_tol=np.array([args.velocity_tol], dtype=np.float32),
        newton_max_iter=np.array([args.newton_max_iter], dtype=np.int32),
        d_hat=np.array([args.d_hat], dtype=np.float32),
        contact_resistance=np.array([args.contact_resistance], dtype=np.float32),
        gel_bottom_bc=np.array([args.gel_bottom_bc], dtype="U16"),
        gel_constraint_strength=np.array([args.gel_constraint_strength], dtype=np.float32),
        indentor_constraint_strength=np.array([args.indentor_constraint_strength], dtype=np.float32),
        indentor_subdiv=np.array([args.indentor_subdiv], dtype=np.int32),
        indentor_geom=np.array([args.indentor_geom], dtype="U16"),
        press_steps=np.array([args.press_steps], dtype=np.int32),
        settle_steps=np.array([args.settle_steps], dtype=np.int32),
        shear_steps=np.array([args.shear_steps], dtype=np.int32),
        shear_settle=np.array([args.shear_settle], dtype=np.int32),
        dt=np.array([args.dt], dtype=np.float32),
        marker_side=np.array([args.marker_side], dtype=np.int32),
        poisson=np.array([args.poisson], dtype=np.float32),
        n_tet_verts=np.array([scalars["n_tet_verts"]], dtype=np.int32),
        marker_sampling=np.array([scalars.get("marker_sampling", "unknown")], dtype="U32"),
        force_marker_sampling=np.array([scalars.get("force_marker_sampling", "unknown")], dtype="U32"),
        load_mode=np.array([LOAD_MODES.index(scalars.get("load_mode", args.load_mode))], dtype=np.int32),
        load_mode_names=np.array(",".join(LOAD_MODES), dtype="U64"),
        meta=np.array(
            f"gt=uipc_ipc_SHEAR; tacex_uipc; disp_units=m; contact_force_units=N/m^2 "
            f"(traction_on_gel_top); contact_force_total_units=N; force_on_gel; "
            f"geom={args.indentor_geom}",
            dtype="U180",
        ),
    )
    if traj is not None:
        payload["disp_traj"] = traj[None, ...].astype(np.float32)
        payload["traj_fracs"] = np.linspace(0.0, 1.0, traj.shape[0]).astype(np.float32)
    cp = contact_profile(coords)
    if cp is not None:
        payload["contact_profile"] = cp[None, ...].astype(np.float32)
        payload["contact_profile_name"] = np.array([args.indentor_mesh or "bolt_hex"], dtype="U64")
    np.savez_compressed(
        os.path.join(out_dir, "uipc_gt_shear.npz"),
        **payload,
    )
    flog(f"  saved field -> {os.path.join(out_dir, 'uipc_gt_shear.npz')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _hard_exit(code=0):
    """Isaac's app.close() can hang; flush + hard-exit like the PhysX driver."""
    flog(f"exit {code}")
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(code)


def main():
    np.random.seed(args.seed)
    coords = marker_grid(args.marker_side)

    # one SimulationContext for the whole process; each run() gets a fresh UipcSim.
    sim_cfg = sim_utils.SimulationCfg(dt=args.dt, gravity=(0.0, 0.0, -9.8), device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    if args.smoke:
        flog(f"SMOKE: gel_res={args.gel_res} eps_velocity={args.eps_velocity}")
        field, force_normal, force_friction, sc, traj = run_one(
            sim, args.gel_res, args.eps_velocity, coords, verbose=True
        )
        print("SMOKE uipc:", "top_markers", field.shape,
              "force_normal", force_normal.shape, "force_friction", force_friction.shape,
              "traj", None if traj is None else traj.shape,
              "load_mode", args.load_mode,
              "peak_uz", round(sc["peak_uz"], 5),
              "max|tang|", round(sc["max_tang"], 5),
              "mean|tang|", round(sc["mean_tang"], 5),
              "verts", sc["n_tet_verts"],
              "t", round(sc["solve_time_s"], 2), flush=True)
        print("SMOKE_UIPC_OK", flush=True)
        _hard_exit(0)

    if args.single:
        flog(f"SINGLE: gel_res={args.gel_res} eps_velocity={args.eps_velocity}")
        field, force_normal, force_friction, sc, traj = run_one(
            sim, args.gel_res, args.eps_velocity, coords, verbose=True
        )
        save_field(args.out, coords, field, force_normal, force_friction, sc, traj)
        print("SINGLE_UIPC_OK", sc, flush=True)
        _hard_exit(0)

    if args.batch:
        # ONE boot, all frames x reps of a combo. Per-frame depth/g/sx/sy and,
        # optionally, R/mu/E come from --batch-rows.  The optional material and
        # geometry columns let a convergence pilot amortise one Isaac boot across
        # heterogeneous production frames without changing the legacy row format.
        # Saves {out}/frame_FFF/rep_R/uipc_gt_shear.npz (sweep layout). Resumable (skip existing).
        rows = []
        with open(args.batch_rows) as fh:
            for line in fh:
                t = line.split()
                if len(t) >= 5:
                    lm = t[5] if len(t) >= 6 else args.load_mode
                    indentor_r = float(t[6]) if len(t) >= 9 else args.indentor_r
                    mu = float(t[7]) if len(t) >= 9 else args.mu
                    youngs = float(t[8]) if len(t) >= 9 else args.youngs
                    rows.append((
                        int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]),
                        lm, indentor_r, mu, youngs,
                    ))
        flog(f"BATCH: {len(rows)} frames x {args.batch_reps} reps, gel_res={args.gel_res} "
             f"eps={args.eps_velocity} R={args.indentor_r} mu={args.mu} E={args.youngs}")
        base_seed = args.seed
        ndone = 0
        for (fi, depth, g, sx, sy, lm, indentor_r, mu, youngs) in rows:
            args.depth = depth                 # mutate globals -> build_schedule()/shear_xy()/save_field read them
            args.drive_ratio = g
            args.shear_x = sx
            args.shear_y = sy
            args.load_mode = lm
            args.indentor_r = indentor_r
            args.mu = mu
            args.youngs = youngs
            for r in range(1, args.batch_reps + 1):
                out = os.path.join(args.out, f"frame_{fi:03d}", f"rep_{r}")
                if os.path.exists(os.path.join(out, "uipc_gt_shear.npz")):
                    ndone += 1; continue
                args.seed = base_seed + 1000 * r
                field, force_normal, force_friction, sc, traj = run_one(
                    sim, args.gel_res, args.eps_velocity, coords, verbose=False
                )
                save_field(out, coords, field, force_normal, force_friction, sc, traj)
                ndone += 1
                flog(f"  batch frame {fi} rep {r}: peak_uz={sc['peak_uz']:.5f} "
                     f"mean_tang={sc['mean_tang']:.5f} solve={sc['solve_time_s']:.1f}s "
                     f"run={sc['run_time_s']:.1f}s done={ndone}")
        print(f"BATCH_UIPC_OK frames={len(rows)} reps={args.batch_reps} saved~{ndone}", flush=True)
        _hard_exit(0)

    # default: convergence sweep over both knobs --------------------------------
    # NOTE: this in-process path creates one UipcSim per setting and is UNVERIFIED
    # (multiple engines + repeated add_physics_callback on one SimulationContext).
    # The trusted path is infra/gen_uipc_convergence.sh (--single, one process per
    # setting) + aggregate_uipc_convergence.py. This block is kept correct but is a
    # convenience only. Mesh is refined by INCREASING gel_res (finest = max).
    res_list = [int(x) for x in args.gel_res_list.split(",") if x.strip()]
    eps_list = [float(x) for x in args.eps_velocity_list.split(",") if x.strip()]
    res_list.sort()           # coarse -> fine (small res -> large res)
    eps_list.sort(reverse=True)  # coarse -> fine (large eps -> small eps)
    flog(f"CONVERGENCE: gel_res in {res_list}  eps_velocity in {eps_list}")

    res_fine = max(res_list)
    eps_fine = min(eps_list)

    fields = {}   # (res, ev) -> field
    rows = []     # scalar rows
    for res in res_list:
        for ev in eps_list:
            flog(f"-- run gel_res={res} eps_velocity={ev}")
            field, force_normal, force_friction, sc, _traj = run_one(
                sim, res, ev, coords, verbose=False
            )
            fields[(res, ev)] = (field, force_normal, force_friction)
            rows.append(sc)
            flog(f"   -> peak_uz={sc['peak_uz']:.5f} mean|tang|={sc['mean_tang']:.5f} "
                 f"verts={sc['n_tet_verts']} t={sc['solve_time_s']:.1f}s")

    # successive-refinement distances (finer level is the reference per axis) ----
    # mesh axis: fix the finest eps, refine gel_res (coarse -> fine)
    mesh_conv = []
    for a, b in zip(res_list[:-1], res_list[1:]):
        d = channel_rel_l2(fields[(a, eps_fine)][0], fields[(b, eps_fine)][0])
        mesh_conv.append({"from_res": a, "to_res": b, **d})
        flog(f"   mesh {a}->{b}: tang={d['tangential']:.4f} norm={d['normal']:.4f}")
    # friction axis: fix the finest mesh, refine eps_velocity (coarse -> fine)
    fric_conv = []
    for a, b in zip(eps_list[:-1], eps_list[1:]):
        d = channel_rel_l2(fields[(res_fine, a)][0], fields[(res_fine, b)][0])
        fric_conv.append({"from_eps": a, "to_eps": b, **d})
        flog(f"   eps  {a}->{b}: tang={d['tangential']:.4f} norm={d['normal']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    import json
    report = {
        "config": {
            "depth": args.depth, "shear": args.shear, "shear_xy": shear_xy(),
            "indentor_r": args.indentor_r,
            "mu": args.mu, "youngs": args.youngs, "poisson": args.poisson,
            "gel": GEL, "d_hat": args.d_hat, "contact_resistance": args.contact_resistance,
            "marker_side": args.marker_side, "dt": args.dt,
        },
        "gel_res_list": res_list,
        "eps_velocity_list": eps_list,
        "scalars": rows,
        "mesh_convergence": mesh_conv,      # tangential should SHRINK (vs PhysX 0.7-0.9)
        "friction_convergence": fric_conv,  # tangential should SHRINK as eps_velocity->0
    }
    rep_path = os.path.join(args.out, "uipc_convergence.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    # also stash the finest field as the candidate GT, PhysX-compatible
    final_field, final_force_normal, final_force_friction = fields[(res_fine, eps_fine)]
    save_field(args.out, coords, final_field, final_force_normal, final_force_friction,
               next(r for r in rows if r["gel_res"] == res_fine and r["eps_velocity"] == eps_fine))
    flog(f"CONVERGENCE done -> {rep_path}")
    print("CONVERGENCE_UIPC_OK", rep_path, flush=True)
    _hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        flog("FATAL: " + repr(e))
        flog(traceback.format_exc())
        _hard_exit(1)
