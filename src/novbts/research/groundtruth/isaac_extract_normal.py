#!/usr/bin/env python3
"""
Ground-truth extractor — PhysX deformable-body FEM (Isaac Sim), no TacEx.

A rigid indentor (sphere/flat) is driven into a deformable gel block; PhysX
solves the FEM elasticity + frictional contact; we read back the simulation-mesh
NODAL positions, sample the displacement field on a regular marker grid over the
gel top surface, and write the same schema as gt_hertz_mindlin:
    params[N,9], coords[M,2], disp[N,M,3], mode[N], meta

Run inside the container (use the derived image with isaaclab core installed):
  docker run --rm --gpus all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$PWD":/work --entrypoint /isaac-sim/python.sh isaac-lab-fem:latest \
    /work/scripts/isaac_extract_groundtruth.py --smoke
  # then a sweep:
  ... isaac_extract_groundtruth.py --frames 200 --marker-side 24 --out /work/data/phase3_gt_fem

This is the SLOW high-fidelity GT: it validates the analytic GT and provides the
real FEM solve-time baseline for RQ3.  Bulk training data stays Hertz-Mindlin.
"""
import argparse
import sys
import glob
import time

# isaaclab editable install isn't picked up in this image; add source dirs.
for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- parse args BEFORE launching the app (AppLauncher consumes argv) ----
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="1 frame, verbose, no save")
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--marker-side", type=int, default=24)
parser.add_argument("--out", default="/work/data/phase3_gt_fem")
parser.add_argument("--settle-steps", type=int, default=25)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# step-by-step progress log to a mounted file (Isaac swallows stdout; this
# pinpoints exactly where a deadlock occurs).
_PROG = "/work/fem_progress.txt"
def flog(msg):
    with open(_PROG, "a") as f:
        f.write(msg + "\n")
        f.flush()

open(_PROG, "w").close()
flog("start: importing AppLauncher")

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
flog("AppLauncher app created")

import os
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg

# ---- gel geometry (metres) ----
GEL = (0.10, 0.10, 0.04)         # x, y, thickness
GEL_TOP_Z = GEL[2]               # rest top surface z (gel base at z=0)
DT = 0.005
MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]
G_STICK, G_PARTIAL, G_FULL = 0.04, 0.48, 1.0


def build_scene(youngs=1.0e5, poisson=0.45, indentor_r=0.02, mu=0.6):
    flog("build_scene: creating SimulationContext")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
    flog("build_scene: SimulationContext OK; creating ground")
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    flog("build_scene: ground OK; building gel cfg")

    gel_cfg = DeformableObjectCfg(
        prim_path="/World/gel",
        spawn=sim_utils.MeshCuboidCfg(
            size=GEL,
            # NOTE: keep the default hexahedral resolution (probe -> 605 nodes,
            # boots fine).  A high resolution (=10) makes deformable cooking
            # extremely slow and looks like a hang during init.
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0, contact_offset=0.001,
                solver_position_iteration_count=16,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=youngs, poissons_ratio=poisson,
                dynamic_friction=mu,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, GEL[2] / 2.0)),
    )
    flog("build_scene: creating DeformableObject (gel)")
    gel = DeformableObject(gel_cfg)
    flog("build_scene: gel DeformableObject OK; building indentor cfg")

    ind_cfg = RigidObjectCfg(
        prim_path="/World/indentor",
        spawn=sim_utils.SphereCfg(
            radius=indentor_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=mu, dynamic_friction=mu),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, GEL_TOP_Z + indentor_r + 0.01)),
    )
    flog("build_scene: creating RigidObject (indentor)")
    ind = RigidObject(ind_cfg)
    flog("build_scene: indentor RigidObject OK; calling sim.reset()")
    sim.reset()
    flog("build_scene: sim.reset() OK -> scene ready")
    return sim, gel, ind


def top_surface_indices(rest_pos, tol=2e-3):
    """Indices of simulation-mesh nodes on the gel top surface."""
    zmax = rest_pos[:, 2].max()
    return np.where(rest_pos[:, 2] > zmax - tol)[0]


def bottom_indices(rest_pos, tol=2e-3):
    zmin = rest_pos[:, 2].min()
    return np.where(rest_pos[:, 2] < zmin + tol)[0]


def marker_grid(side):
    xs = np.linspace(-GEL[0] / 2 * 0.9, GEL[0] / 2 * 0.9, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def sample_to_markers(top_rest_xy, top_disp, coords):
    """Nearest-node displacement sampling onto the marker grid."""
    from scipy.spatial import cKDTree
    tree = cKDTree(top_rest_xy)
    _, idx = tree.query(coords, k=1)
    return top_disp[idx]


def run_frame(sim, gel, ind, depth, shear_x, shear_y, indentor_r, settle, verbose=False):
    """
    NORMAL-ONLY FEM frame: lower the indentor by `depth`, settle, read disp.

    Phase-2 (lateral shear of a kinematic rigid body in deep deformable contact)
    deadlocks PhysX on this setup, so tangential/slip stays on the analytic
    Cattaneo-Mindlin model.  This produces a real finite-thickness FEM normal
    field (the genuine improvement over the Hertz half-space) + the slow-solver
    time for RQ3.  shear_x/shear_y are accepted but ignored (kept for schema).
    """
    rest_state = gel.data.default_nodal_state_w.clone()
    gel.write_nodal_state_to_sim(rest_state)
    rest_pos = gel.data.nodal_pos_w[0].cpu().numpy().copy()

    top = top_surface_indices(rest_pos)
    bot = bottom_indices(rest_pos)

    # pin bottom nodes: flag 1.0 == kinematically driven (fixed to target);
    # 0.0 == free/simulated.  (Earlier 0.0 left the gel un-anchored -> it got
    # pushed around, giving unphysical |uz| > indentation depth.)
    kin = gel.data.nodal_kinematic_target.clone()
    kin[0, bot, :3] = torch.tensor(rest_pos[bot], device=kin.device)
    kin[0, bot, 3] = 1.0
    gel.write_nodal_kinematic_target_to_sim(kin)

    cx, cy = 0.0, 0.0
    z0 = GEL_TOP_Z + indentor_r
    steps_down = 15

    def set_indentor(x, y, z):
        pose = ind.data.default_root_state.clone()
        pose[0, 0:3] = torch.tensor([x, y, z], device=pose.device)
        ind.write_root_pose_to_sim(pose[:, :7])

    if verbose:
        flog("run_frame: pinned bottom; lowering (normal-only)")
    for k in range(steps_down):
        f = (k + 1) / steps_down
        set_indentor(cx, cy, z0 - depth * f)
        sim.step(); gel.update(DT); ind.update(DT)
        if verbose and (k % 5 == 0 or k == steps_down - 1):
            flog(f"run_frame: lower step {k+1}/{steps_down}")
    if verbose:
        flog("run_frame: lowering done; settling")
    for s in range(settle):
        sim.step(); gel.update(DT); ind.update(DT)
        if verbose and s % 10 == 0:
            flog(f"run_frame: settle step {s+1}/{settle}")

    pos = gel.data.nodal_pos_w[0].cpu().numpy()
    disp = pos - rest_pos
    if verbose:
        flog("run_frame: read nodal disp; done")
    return rest_pos, disp, top


def label_mode(shear_mag, mu):
    g = shear_mag / max(mu, 1e-6)
    if g < G_STICK: return 0
    if g < G_PARTIAL: return 1
    if g < G_FULL: return 2
    return 3


def main():
    rng = np.random.default_rng(args.seed)
    indentor_r, mu = 0.02, 0.6
    sim, gel, ind = build_scene(indentor_r=indentor_r, mu=mu)

    if args.smoke:
        t0 = time.perf_counter()
        rest, disp, top = run_frame(sim, gel, ind, depth=0.008,
                                    shear_x=0.0, shear_y=0.0,
                                    indentor_r=indentor_r, settle=args.settle_steps,
                                    verbose=True)
        dt = time.perf_counter() - t0
        print("SMOKE nodal:", rest.shape, "top nodes:", top.shape[0],
              "max|disp|:", float(np.abs(disp).max()),
              "peak uz:", float(disp[top, 2].min()), "frame_time_s:", round(dt, 3))
        print("SMOKE_OK")
        simulation_app.close()
        return

    coords = marker_grid(args.marker_side)
    params, disps, modes = [], [], []
    solve_times = []
    for i in range(args.frames):
        depth = rng.uniform(0.003, 0.012)   # normal indentation only
        t0 = time.perf_counter()
        rest, disp, top = run_frame(sim, gel, ind, depth, 0.0, 0.0,
                                    indentor_r, args.settle_steps)
        solve_times.append(time.perf_counter() - t0)
        top_xy = rest[top, :2]
        m_disp = sample_to_markers(top_xy, disp[top], coords)
        params.append([0.0, 0.0, depth, indentor_r, 0.0, 0.0, mu, 1.0, 0.0])
        disps.append(m_disp.astype(np.float32))
        modes.append(0)  # normal mode (FEM shear/slip = open problem, see report)
        flog(f"sweep {i+1}/{args.frames} avg_solve={np.mean(solve_times):.3f}s")
        # incremental save every frame (app.close() hangs in headless, so we
        # must be killable at any point WITHOUT losing data)
        if (i + 1) % 5 == 0 or i == args.frames - 1:
            os.makedirs(args.out, exist_ok=True)
            np.savez_compressed(
                os.path.join(args.out, "fem_gt.npz"),
                params=np.array(params, dtype=np.float32),
                coords=coords.astype(np.float32),
                disp=np.stack(disps).astype(np.float32),
                mode=np.array(modes, dtype=np.int32),
                solve_time_s=np.array(solve_times, dtype=np.float32),
                meta=np.array("gt=physx_deformable_fem_normal_only; isaac-lab-fem; units=m",
                              dtype="U80"),
            )
            flog(f"sweep: saved {i+1} frames to fem_gt.npz")
    print(f"SAVED {args.frames} frames -> {args.out}/fem_gt.npz")
    print(f"FEM solve time: mean={np.mean(solve_times):.3f}s  "
          f"=> {1.0/np.mean(solve_times):.2f} frames/s (the SLOW solver baseline for RQ3)")
    simulation_app.close()


if __name__ == "__main__":
    main()
