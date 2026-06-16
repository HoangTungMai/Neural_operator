#!/usr/bin/env python3
"""
SHEAR ground-truth attempt — PhysX deformable FEM with tangential loading.

The previous extractor (isaac_extract_groundtruth.py) is NORMAL-ONLY because
dragging a deeply-embedded *kinematic* (infinitely stiff) rigid indentor
laterally over-constrains the deformable solver and it deadlocks.

This script attacks the open problem with three changes vs that setup:
  1. SHALLOW indentation for shear frames (light contact, not deep) — the
     deadlock was "deep contact + lateral".
  2. Lateral motion in MANY tiny increments (~velocity control of the kinematic
     body) with settle substeps each — lets the FEM solver track the contact.
  3. Higher solver_position_iteration_count + per-micro-step logging so we see
     EXACTLY where it stalls (or that it now completes).

--smoke runs ONE shear frame, verbose, no save, so we can diagnose fast.
Every lateral micro-step is logged to /work/fem_progress.txt (Isaac swallows
stdout; the file is the live progress signal — DO NOT watch stdout).
"""
import argparse
import sys
import glob
import time

for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="1 shear frame, verbose, no save")
parser.add_argument("--frames", type=int, default=40)
parser.add_argument("--marker-side", type=int, default=24)
parser.add_argument("--out", default="/work/data/phase3_gt_fem_shear")
parser.add_argument("--depth", type=float, default=0.005, help="shallow normal indentation (m)")
parser.add_argument("--shear", type=float, default=0.004, help="lateral travel (m) for smoke")
parser.add_argument("--lower-steps", type=int, default=20)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--lateral-steps", type=int, default=60, help="tiny lateral increments")
parser.add_argument("--lateral-settle", type=int, default=3, help="settle substeps per increment")
parser.add_argument("--solver-iters", type=int, default=30)
parser.add_argument("--hex-res", type=int, default=-1, help="simulation_hexahedral_resolution; -1=default(coarse)")
parser.add_argument("--gel-xy", type=float, default=0.10, help="gel footprint x=y (m)")
parser.add_argument("--gel-z", type=float, default=0.04, help="gel thickness (m)")
parser.add_argument("--indentor-r", type=float, default=0.02, help="sphere indentor radius (m)")
parser.add_argument("--mu", type=float, default=0.6, help="friction coeff (gel+indentor); swept for param coverage")
parser.add_argument("--youngs", type=float, default=1.0e5, help="gel Young's modulus E (Pa); swept for param coverage")
parser.add_argument("--contact-offset", type=float, default=0.002, help="PhysX contact offset (m); scale to gel")
parser.add_argument("--dt", type=float, default=0.005, help="sim timestep (s); small gels need small dt (stability ~ size)")
parser.add_argument("--damping", type=float, default=-1.0, help="vertex_velocity_damping; -1=default. High value suppresses small-gel oscillation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gmin", type=float, default=0.0, help="lower bound of sampled drive ratio g")
parser.add_argument("--gmax", type=float, default=1.3, help="upper bound of sampled drive ratio g; set =gmin=0 for pure-normal (mode 0) frames")
parser.add_argument("--save-trajectory", action="store_true", help="store the lateral loading PATH as per-step snapshots (temporal/loading-history GT)")
parser.add_argument("--traj-steps", type=int, default=8, help="number of trajectory snapshots incl normal (f=0) and final (f=1)")
parser.add_argument("--load-mode", choices=["linear", "ortho", "reverse"], default="linear",
                    help="lateral loading-path shape; same endpoint, different PATH (for path-dependence)")
parser.add_argument("--mix-loads", action="store_true", help="randomly pick load-mode per frame (linear/ortho/reverse) -> path-dependent dataset")
args = parser.parse_args()

_PROG = "/work/fem_progress.txt"
def flog(msg):
    with open(_PROG, "a") as f:
        f.write(msg + "\n"); f.flush()

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

GEL = (args.gel_xy, args.gel_xy, args.gel_z)
GEL_TOP_Z = GEL[2]
DT = args.dt
MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]
G_STICK, G_PARTIAL, G_FULL = 0.04, 0.48, 1.0


def build_scene(youngs=1.0e5, poisson=0.45, indentor_r=0.02, mu=0.6):
    flog("build_scene: SimulationContext")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    gel_cfg = DeformableObjectCfg(
        prim_path="/World/gel",
        spawn=sim_utils.MeshCuboidCfg(
            size=GEL,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0, contact_offset=args.contact_offset,
                solver_position_iteration_count=args.solver_iters,
                **({"simulation_hexahedral_resolution": args.hex_res} if args.hex_res > 0 else {}),
                **({"vertex_velocity_damping": args.damping} if args.damping >= 0 else {}),
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=youngs, poissons_ratio=poisson, dynamic_friction=mu,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, GEL[2] / 2.0)),
    )
    gel = DeformableObject(gel_cfg)
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
    ind = RigidObject(ind_cfg)
    flog("build_scene: sim.reset()")
    sim.reset()
    flog("build_scene: ready")
    return sim, gel, ind


def _z_tol(rest_pos):
    """Surface-layer tolerance: a fraction of gel thickness, not absolute (the
    old 2mm was as thick as a 2mm gel -> selected every node)."""
    return max((rest_pos[:, 2].max() - rest_pos[:, 2].min()) * 0.08, 1e-5)


def top_surface_indices(rest_pos, tol=None):
    tol = _z_tol(rest_pos) if tol is None else tol
    zmax = rest_pos[:, 2].max()
    return np.where(rest_pos[:, 2] > zmax - tol)[0]


def bottom_indices(rest_pos, tol=None):
    tol = _z_tol(rest_pos) if tol is None else tol
    zmin = rest_pos[:, 2].min()
    return np.where(rest_pos[:, 2] < zmin + tol)[0]


def marker_grid(side):
    xs = np.linspace(-GEL[0] / 2 * 0.9, GEL[0] / 2 * 0.9, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def sample_to_markers(top_rest_xy, top_disp, coords):
    from scipy.spatial import cKDTree
    tree = cKDTree(top_rest_xy)
    _, idx = tree.query(coords, k=1)
    return top_disp[idx]


LOAD_MODES = ["linear", "ortho", "reverse"]


def path_xy(f, sx, sy, mode):
    """Lateral indentor offset at drag fraction f in [0,1]. All modes share the
    SAME endpoint (sx,sy) at f=1 but take different PATHS, so the final state is
    genuinely path-dependent (friction hysteresis) -> a real loading-history test.
      linear : straight ramp 0 -> (sx,sy)
      ortho  : L-turn (drag x to full, then y) -> history matters at the corner
      reverse: overshoot to 1.5x then unload back to (sx,sy) -> load-unload hysteresis
    """
    if mode == "ortho":
        if f <= 0.5:
            return sx * (f / 0.5), 0.0
        return sx, sy * ((f - 0.5) / 0.5)
    if mode == "reverse":
        s = 1.5 * (f / 0.66) if f <= 0.66 else 1.5 - 0.5 * ((f - 0.66) / 0.34)
        return sx * s, sy * s
    return sx * f, sy * f


def run_shear_frame(sim, gel, ind, depth, shear_x, shear_y, indentor_r, verbose=False,
                    load_mode="linear", traj_steps=0):
    """Normal indent (shallow) -> settle -> LATERAL drag in tiny increments.

    Returns (rest_pos, disp_final, top_idx, traj) where traj is a list of nodal
    displacement snapshots [n_nodes,3] along the loading path (empty if traj_steps==0;
    traj[0] is the pure-normal state at f=0, traj[-1] the final state at f=1).
    Each lateral micro-step is logged; if a step takes pathologically long the
    caller's watchdog (fem_progress.txt stalls) catches the deadlock.
    """
    rest_state = gel.data.default_nodal_state_w.clone()
    gel.write_nodal_state_to_sim(rest_state)
    rest_pos = gel.data.nodal_pos_w[0].cpu().numpy().copy()
    top = top_surface_indices(rest_pos)
    bot = bottom_indices(rest_pos)

    kin = gel.data.nodal_kinematic_target.clone()
    kin[0, bot, :3] = torch.tensor(rest_pos[bot], device=kin.device)
    kin[0, bot, 3] = 1.0
    gel.write_nodal_kinematic_target_to_sim(kin)

    cx, cy = 0.0, 0.0
    z0 = GEL_TOP_Z + indentor_r
    z_target = z0 - depth

    def set_indentor(x, y, z):
        pose = ind.data.default_root_state.clone()
        pose[0, 0:3] = torch.tensor([x, y, z], device=pose.device)
        ind.write_root_pose_to_sim(pose[:, :7])

    # --- phase 1: normal indent (shallow) ---
    if verbose:
        flog(f"normal: lowering to depth={depth} over {args.lower_steps} steps")
    for k in range(args.lower_steps):
        f = (k + 1) / args.lower_steps
        set_indentor(cx, cy, z0 - depth * f)
        sim.step(); gel.update(DT); ind.update(DT)
    for s in range(args.settle_steps):
        sim.step(); gel.update(DT); ind.update(DT)
    if verbose:
        pos = gel.data.nodal_pos_w[0].cpu().numpy()
        flog(f"normal settled: peak uz={float((pos-rest_pos)[top,2].min()):.5f}")

    # --- trajectory snapshots: f=0 (normal) + evenly-spaced fractions to f=1 ---
    snaps = list(np.linspace(0.0, 1.0, traj_steps)) if traj_steps > 0 else []
    traj, si = [], 0
    if snaps and snaps[0] <= 1e-9:
        traj.append((gel.data.nodal_pos_w[0].cpu().numpy() - rest_pos).copy()); si = 1

    # --- phase 2: lateral drag in tiny increments (~velocity control) ---
    nlat = args.lateral_steps
    if verbose:
        flog(f"shear[{load_mode}]: dragging to ({shear_x:.4f},{shear_y:.4f}) over {nlat} micro-steps")
    for k in range(nlat):
        f = (k + 1) / nlat
        t_step = time.perf_counter()
        wx, wy = path_xy(f, shear_x, shear_y, load_mode)
        set_indentor(cx + wx, cy + wy, z_target)
        sim.step(); gel.update(DT); ind.update(DT)
        for _ in range(args.lateral_settle):
            sim.step(); gel.update(DT); ind.update(DT)
        while si < len(snaps) and f + 1e-9 >= snaps[si]:
            traj.append((gel.data.nodal_pos_w[0].cpu().numpy() - rest_pos).copy()); si += 1
        if verbose and (k % 5 == 0 or k == nlat - 1):
            dtk = time.perf_counter() - t_step
            flog(f"shear micro-step {k+1}/{nlat}  step_time={dtk:.3f}s")

    pos = gel.data.nodal_pos_w[0].cpu().numpy()
    disp = pos - rest_pos
    if verbose:
        tang = np.linalg.norm(disp[top, :2], axis=1).max()
        flog(f"shear done: max|tang|={tang:.5f}  peak uz={disp[top,2].min():.5f}")
    return rest_pos, disp, top, traj


def label_mode(shear_mag, mu):
    g = shear_mag / max(mu, 1e-6)
    if g < G_STICK: return 0
    if g < G_PARTIAL: return 1
    if g < G_FULL: return 2
    return 3


def main():
    rng = np.random.default_rng(args.seed)
    indentor_r, mu = args.indentor_r, args.mu
    sim, gel, ind = build_scene(youngs=args.youngs, indentor_r=indentor_r, mu=mu)

    if args.smoke:
        t0 = time.perf_counter()
        rest, disp, top, traj = run_shear_frame(sim, gel, ind, depth=args.depth,
                                          shear_x=args.shear, shear_y=0.0,
                                          indentor_r=indentor_r, verbose=True,
                                          load_mode=args.load_mode,
                                          traj_steps=args.traj_steps if args.save_trajectory else 0)
        dt = time.perf_counter() - t0
        if traj:
            print("SMOKE traj frames:", len(traj), "load_mode:", args.load_mode)
        tnorm = np.linalg.norm(disp[top, :2], axis=1)
        tang = float(tnorm.max())
        mean_tang = float(tnorm.mean())          # integral-ish: converges better than pointwise max
        peak_uz = float(disp[top, 2].min())
        print("SMOKE shear nodal:", rest.shape, "top:", top.shape[0],
              "max|tang|:", round(tang, 5), "mean|tang|:", round(mean_tang, 5),
              "peak uz:", round(peak_uz, 5), "frame_time_s:", round(dt, 2))
        print("SMOKE_SHEAR_OK")
        # convergence log: append one line (hard-exit below avoids app.close() hang)
        with open("/work/convergence.txt", "a") as f:
            f.write(f"hex_res={args.hex_res} gel_xy={args.gel_xy} gel_z={args.gel_z} "
                    f"ind_r={args.indentor_r} nodes={rest.shape[0]} top={top.shape[0]} "
                    f"depth={args.depth} shear={args.shear} max_tang={tang:.6f} "
                    f"mean_tang={mean_tang:.6f} peak_uz={peak_uz:.6f} frame_s={dt:.2f}\n")
            f.flush(); os.fsync(f.fileno())
        flog("smoke result appended to convergence.txt; hard exit")
        os._exit(0)   # app.close() hangs in headless; exit immediately, data is saved

    coords = marker_grid(args.marker_side)
    params, disps, modes, solve_times = [], [], [], []
    disps_traj, load_mode_ids = [], []
    traj_steps = args.traj_steps if args.save_trajectory else 0
    for i in range(args.frames):
        depth = rng.uniform(0.004, 0.007)
        # drive ratio g spread across stick/partial/full via shear magnitude
        # (gmin==gmax==0 -> pure-normal frames, mode 0, to balance the slip class)
        g = rng.uniform(args.gmin, args.gmax)
        shear_mag = g * mu * 0.01            # scale lateral travel (m) by drive ratio
        theta = rng.uniform(0, 2 * np.pi)
        sx, sy = shear_mag * np.cos(theta), shear_mag * np.sin(theta)
        lm = LOAD_MODES[int(rng.integers(len(LOAD_MODES)))] if args.mix_loads else args.load_mode
        t0 = time.perf_counter()
        rest, disp, top, traj = run_shear_frame(sim, gel, ind, depth, sx, sy, indentor_r,
                                                load_mode=lm, traj_steps=traj_steps)
        solve_times.append(time.perf_counter() - t0)
        m_disp = sample_to_markers(rest[top, :2], disp[top], coords)
        params.append([0.0, 0.0, depth, indentor_r, sx, sy, mu, args.youngs, 0.0])
        disps.append(m_disp.astype(np.float32))
        if traj_steps > 0:
            tm = np.stack([sample_to_markers(rest[top, :2], tf[top], coords) for tf in traj])
            disps_traj.append(tm.astype(np.float32))
            load_mode_ids.append(LOAD_MODES.index(lm))
        # mode from the SAMPLED drive ratio g (standard Cattaneo-Mindlin
        # thresholds); shear_mag is a lateral TRAVEL in metres, not a force
        # ratio, so do NOT pass it to label_mode as g.
        modes.append(label_mode(g * mu, mu))
        flog(f"sweep {i+1}/{args.frames} g={g:.2f} avg_solve={np.mean(solve_times):.2f}s")
        if (i + 1) % 5 == 0 or i == args.frames - 1:
            os.makedirs(args.out, exist_ok=True)
            np.savez_compressed(
                os.path.join(args.out, "fem_gt_shear.npz"),
                params=np.array(params, dtype=np.float32),
                coords=coords.astype(np.float32),
                disp=np.stack(disps).astype(np.float32),
                mode=np.array(modes, dtype=np.int32),
                solve_time_s=np.array(solve_times, dtype=np.float32),
                meta=np.array("gt=physx_deformable_fem_SHEAR; isaac-lab-fem; units=m", dtype="U80"),
                **({"disp_traj": np.stack(disps_traj).astype(np.float32),
                    "load_mode": np.array(load_mode_ids, dtype=np.int32),
                    "traj_fracs": np.linspace(0.0, 1.0, args.traj_steps).astype(np.float32),
                    "load_mode_names": np.array(",".join(LOAD_MODES))} if traj_steps > 0 else {}),
            )
            flog(f"saved {i+1} frames")
    print(f"SAVED {args.frames} shear frames -> {args.out}/fem_gt_shear.npz")
    print(f"FEM solve time mean={np.mean(solve_times):.2f}s => {1.0/np.mean(solve_times):.2f} fps")
    flog(f"SWEEP DONE {args.frames} frames; hard exit")
    os._exit(0)   # app.close() hangs in headless; data already saved incrementally


if __name__ == "__main__":
    main()
