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
fields (``gel_res, eps_velocity, d_hat, contact_resistance, n_tet_verts``).
Coordinate convention is Z-up (gel top face at z=GEL_Z, uz is the normal
component) — identical to the PhysX driver, NOT the Y-up libuipc-samples.

HOW IT RUNS
-----------
Inside the ``isaac-lab-tacex:latest`` container, like the PhysX driver:

    /isaac-sim/python.sh -m novbts.groundtruth.tacex_uipc_extract_shear --smoke

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
parser.add_argument("--batch-rows", default="",
                    help="path to a whitespace table, one row per frame: 'frame_idx depth g sx sy'")
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
parser.add_argument("--mu", type=float, default=0.6, help="friction coeff (-> default_friction_ratio)")
parser.add_argument("--youngs", type=float, default=1.0e5, help="gel Young's modulus E (Pa)")
parser.add_argument("--poisson", type=float, default=0.45, help="gel Poisson ratio")
parser.add_argument("--gel-density", type=float, default=1.0e3, help="gel mass density (kg/m^3)")
parser.add_argument("--indentor-youngs", type=float, default=5.0e8,
                    help="indentor Young's modulus (Pa); high => effectively rigid")
# --- stepping schedule ------------------------------------------------------
parser.add_argument("--press-steps", type=int, default=40, help="frames to lower the indentor")
parser.add_argument("--settle-steps", type=int, default=10, help="frames to settle after press")
parser.add_argument("--shear-steps", type=int, default=80, help="frames for the lateral drag")
parser.add_argument("--shear-settle", type=int, default=10, help="frames to settle after shear")
parser.add_argument("--dt", type=float, default=0.01, help="UIPC timestep (s)")
# --- IPC convergence knobs --------------------------------------------------
parser.add_argument("--gel-res", type=int, default=12,
                    help="gel structured-mesh resolution (cells/footprint-axis); MESH refinement knob")
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

import isaaclab.sim as sim_utils

from tacex_uipc import UipcSim, UipcSimCfg

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
GEOM_CODE = {"sphere": 0, "flat": 1, "cylinder": 2, "mesh": 3}


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


def structured_tet_box(size, gel_res):
    """Deterministic structured tet mesh of the gel box -> (points Nx3, tets Mx4).

    A regular nx*ny*nz grid of hexes, each split into 6 tets (positive volume).
    Used INSTEAD of wildmeshing for the gel because a convergence study needs:
      * determinism  — same gel_res -> identical mesh (so the friction-axis sweep
        isolates eps_velocity instead of conflating it with random remeshing);
      * a FLAT regular top face — so top-face detection and marker sampling are
        well-posed (wildmeshing's irregular top collapsed the sampled field to a
        single vertex at fine resolution);
      * monotone refinement — gel_res up => strictly finer => clean limit.

    gel_res = number of cells along each footprint axis (x=y); z cells scale to
    keep cells roughly cubic.
    """
    lx, ly, lz = size
    nx = ny = int(gel_res)
    cell = lx / nx
    nz = max(1, int(round(lz / cell)))
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
        sched.append((sx * f, sy * f, -z_down))
    for _ in range(n_shear_settle):
        sched.append((sx, sy, -z_down))
    return sched, gap


def shear_xy():
    """Return the lateral endpoint vector.

    ``--shear`` remains the original +x shorthand. Paired PhysX comparisons use
    ``--shear-x/--shear-y`` so the IPC load matches one sampled PhysX frame.
    """
    if args.shear_x is None:
        return float(args.shear), 0.0
    return float(args.shear_x), float(0.0 if args.shear_y is None else args.shear_y)


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
    gel_pts, gel_tets = structured_tet_box(GEL, gel_res)
    flog(f"  gel tets: {gel_pts.shape[0]} verts, {gel_tets.shape[0]} tets")
    gel_mesh = to_uipc_mesh(gel_pts, gel_tets)
    # youngs_poisson takes Young's modulus in SI Pa (TacEx passes ``youngs*MPa``,
    # i.e. an already-SI value); args.youngs is already in Pa, so pass it raw.
    moduli = ElasticModuli.youngs_poisson(args.youngs, args.poisson)
    snh.apply_to(gel_mesh, moduli, mass_density=args.gel_density)
    default_element.apply_to(gel_mesh)
    # Dirichlet BC: pin the bottom face (z = 0) by constraining those vertices.
    spc.apply_to(gel_mesh, 100.0)  # constraint strength ratio
    gel_pos0 = gel_mesh.positions().view().reshape(-1, 3).copy()
    # structured mesh => bottom verts sit exactly at z=0; a tight tol is safe.
    gel_bottom_mask = gel_pos0[:, 2] <= (gel_pos0[:, 2].min() + 1e-6)
    gel_object = scene.objects().create("gel")
    gel_slot, _ = gel_object.geometries().create(gel_mesh)

    # --- indentor (stiff SNH sphere, all vertices prescribed => kinematic) ---
    # Indentor stays on wildmeshing: it is rigid+driven, so its interior tets are
    # irrelevant and the contact surface is the deterministic icosphere; its mesh
    # is regenerated each run but does not enter the gel-field convergence metric.
    flog("  meshing indentor sphere (deterministic fan) ...")
    c0 = (0.0, 0.0, GEL_TOP_Z + args.indentor_r + args.d_hat)
    ind_V, ind_F = icosphere_surface(args.indentor_r, subdiv=args.indentor_subdiv, center=c0)
    ind_pts, ind_tets = fan_tet_sphere(ind_V, ind_F, c0)
    flog(f"  indentor tets: {ind_pts.shape[0]} verts, {ind_tets.shape[0]} tets")
    ind_mesh = to_uipc_mesh(ind_pts, ind_tets)
    ind_moduli = ElasticModuli.youngs_poisson(args.indentor_youngs, 0.45)
    snh.apply_to(ind_mesh, ind_moduli, mass_density=args.gel_density)
    default_element.apply_to(ind_mesh)
    spc.apply_to(ind_mesh, 100.0)
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


def run_one(sim, gel_res, eps_velocity, coords, verbose=False):
    """Run a single (gel_res, eps_velocity) indent+shear; return field + scalars."""
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
    for step in range(n_total):
        uipc_sim.step()
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

    tnorm = np.linalg.norm(top_markers[:, :2], axis=1)
    scalars = {
        "gel_res": int(gel_res),
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
    return top_markers.astype(np.float32), scalars


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
def save_field(out_dir, coords, field, scalars):
    os.makedirs(out_dir, exist_ok=True)
    sx, sy = shear_xy()
    # params row layout matches isaac_extract_shear.py:
    # [cx, cy, depth, R, sx, sy, mu, youngs, geom_code]
    params = np.array([[0.0, 0.0, args.depth, args.indentor_r, sx, sy,
                        args.mu, args.youngs, float(GEOM_CODE["sphere"])]], dtype=np.float32)
    # mode is the Cattaneo-Mindlin slip class. The PhysX sweep labels it from the
    # SAMPLED drive ratio g. For ad-hoc --single runs the shear is just a fixed
    # lateral TRAVEL in metres, so a faithful g is unavailable -> store -1. Scripted
    # production sweeps pass --drive-ratio explicitly, so their labels match PhysX.
    mode_value = -1 if args.drive_ratio is None else label_mode(float(args.drive_ratio) * args.mu, args.mu)
    mode = np.array([mode_value], dtype=np.int32)
    np.savez_compressed(
        os.path.join(out_dir, "uipc_gt_shear.npz"),
        params=params,
        coords=coords.astype(np.float32),
        disp=field[None, ...].astype(np.float32),  # (1, M, 3)
        mode=mode,
        solve_time_s=np.array([scalars["solve_time_s"]], dtype=np.float32),
        gel_res=np.array([scalars["gel_res"]], dtype=np.int32),
        eps_velocity=np.array([scalars["eps_velocity"]], dtype=np.float32),
        velocity_tol=np.array([args.velocity_tol], dtype=np.float32),
        d_hat=np.array([args.d_hat], dtype=np.float32),
        contact_resistance=np.array([args.contact_resistance], dtype=np.float32),
        n_tet_verts=np.array([scalars["n_tet_verts"]], dtype=np.int32),
        marker_sampling=np.array([scalars.get("marker_sampling", "unknown")], dtype="U32"),
        meta=np.array("gt=uipc_ipc_SHEAR; tacex_uipc; units=m; geom=sphere", dtype="U96"),
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
        field, sc = run_one(sim, args.gel_res, args.eps_velocity, coords, verbose=True)
        print("SMOKE uipc:", "top_markers", field.shape,
              "peak_uz", round(sc["peak_uz"], 5),
              "max|tang|", round(sc["max_tang"], 5),
              "mean|tang|", round(sc["mean_tang"], 5),
              "verts", sc["n_tet_verts"],
              "t", round(sc["solve_time_s"], 2), flush=True)
        print("SMOKE_UIPC_OK", flush=True)
        _hard_exit(0)

    if args.single:
        flog(f"SINGLE: gel_res={args.gel_res} eps_velocity={args.eps_velocity}")
        field, sc = run_one(sim, args.gel_res, args.eps_velocity, coords, verbose=True)
        save_field(args.out, coords, field, sc)
        print("SINGLE_UIPC_OK", sc, flush=True)
        _hard_exit(0)

    if args.batch:
        # ONE boot, all frames x reps of a combo. Per-frame depth/g/sx/sy from
        # --batch-rows (mirrors gen_uipc_sweep.sh sampling). Fixed combo R/mu/E/gel_res/eps.
        # Saves {out}/frame_FFF/rep_R/uipc_gt_shear.npz (sweep layout). Resumable (skip existing).
        rows = []
        with open(args.batch_rows) as fh:
            for line in fh:
                t = line.split()
                if len(t) >= 5:
                    rows.append((int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])))
        flog(f"BATCH: {len(rows)} frames x {args.batch_reps} reps, gel_res={args.gel_res} "
             f"eps={args.eps_velocity} R={args.indentor_r} mu={args.mu} E={args.youngs}")
        base_seed = args.seed
        ndone = 0
        for (fi, depth, g, sx, sy) in rows:
            args.depth = depth                 # mutate globals -> build_schedule()/shear_xy()/save_field read them
            args.drive_ratio = g
            args.shear_x = sx
            args.shear_y = sy
            for r in range(1, args.batch_reps + 1):
                out = os.path.join(args.out, f"frame_{fi:03d}", f"rep_{r}")
                if os.path.exists(os.path.join(out, "uipc_gt_shear.npz")):
                    ndone += 1; continue
                args.seed = base_seed + 1000 * r
                field, sc = run_one(sim, args.gel_res, args.eps_velocity, coords, verbose=False)
                save_field(out, coords, field, sc)
                ndone += 1
                flog(f"  batch frame {fi} rep {r}: peak_uz={sc['peak_uz']:.5f} "
                     f"mean_tang={sc['mean_tang']:.5f} t={sc['solve_time_s']:.1f}s done={ndone}")
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
            field, sc = run_one(sim, res, ev, coords, verbose=False)
            fields[(res, ev)] = field
            rows.append(sc)
            flog(f"   -> peak_uz={sc['peak_uz']:.5f} mean|tang|={sc['mean_tang']:.5f} "
                 f"verts={sc['n_tet_verts']} t={sc['solve_time_s']:.1f}s")

    # successive-refinement distances (finer level is the reference per axis) ----
    # mesh axis: fix the finest eps, refine gel_res (coarse -> fine)
    mesh_conv = []
    for a, b in zip(res_list[:-1], res_list[1:]):
        d = channel_rel_l2(fields[(a, eps_fine)], fields[(b, eps_fine)])
        mesh_conv.append({"from_res": a, "to_res": b, **d})
        flog(f"   mesh {a}->{b}: tang={d['tangential']:.4f} norm={d['normal']:.4f}")
    # friction axis: fix the finest mesh, refine eps_velocity (coarse -> fine)
    fric_conv = []
    for a, b in zip(eps_list[:-1], eps_list[1:]):
        d = channel_rel_l2(fields[(res_fine, a)], fields[(res_fine, b)])
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
    save_field(args.out, coords, fields[(res_fine, eps_fine)],
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
