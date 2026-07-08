#!/usr/bin/env python3
"""Container-only grasp/press demo driver.

This driver keeps Isaac startup and asset spawning in the loop, while the tactile
rendering path is the same host-testable VBTSSensor used by unit smoke tests.
The robot control is intentionally manifest-driven and open-loop so P0/P3 can
pin real joint names before later IK tuning.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

if "/work/src" not in sys.path:
    sys.path.append("/work/src")
for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from novbts.sim.gel_contact import (
    ShearTracker,
    T_inv,
    gel_grid,
    pen_cuboid,
    pen_cylinder,
    pen_sphere,
    pose_to_T,
)
from novbts.sim.vbts_sensor import VBTSSensor


PROGRESS = "/work/sim_grasp_demo_progress.txt"


def flog(msg: str) -> None:
    Path(PROGRESS).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def _save_png(img: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.0, 2.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    ax.set_axis_off()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _pen_for_object(kind: str, size: float, T_gel_obj: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(T_gel_obj)):
        side = int(round(np.sqrt(grid.shape[0])))
        return np.zeros((side, side), dtype=np.float32)
    if kind == "sphere":
        return pen_sphere(T_gel_obj, size / 2.0, grid)
    if kind == "cube":
        return pen_cuboid(T_gel_obj, (size, size, size), grid)
    if kind == "cylinder":
        return pen_cylinder(T_gel_obj, (size / 2.0, size), grid)
    raise ValueError(kind)


def _quat_wxyz_from_R(R: np.ndarray) -> list[float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def _Rx(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _Ry(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _Rz(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_from_rpy_deg(rpy: str) -> list[float]:
    vals = [float(x.strip()) for x in rpy.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--robot-root-rpy-deg must be 'roll,pitch,yaw'")
    roll, pitch, yaw = [np.deg2rad(v) for v in vals]
    R = _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)
    return _quat_wxyz_from_R(R)


def _vec3_arg(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("expected comma-separated x,y,z")
    return np.asarray(vals, dtype=np.float64)


def _gel_R_from_normal(normal_w: np.ndarray) -> np.ndarray:
    z = np.asarray(normal_w, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1.0e-12)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1.0e-12)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _try_start_isaac(manifest_path: str, args) -> dict:
    """Spawn a minimal scene. Failures are logged but do not hide tactile output."""
    physics_probe = {"enabled": False}
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        _simulation_app = app_launcher.app
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg

        sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=600.0).func("/World/light", sim_utils.DomeLightCfg(intensity=600.0))
        gripper_joint_expr = (
            [
                "finger_joint",
                "right_outer_knuckle_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "left_inner_finger_knuckle_joint",
                "right_inner_finger_knuckle_joint",
            ]
            if args.gripper_control_mode == "all"
            else ["finger_joint"]
        )
        passive_gripper_joint_expr = [
            "right_outer_knuckle_joint",
            "left_outer_finger_joint",
            "right_outer_finger_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
            "left_inner_finger_knuckle_joint",
            "right_inner_finger_knuckle_joint",
        ]
        robot_rigid_kwargs = {}
        if args.robot_solver_position_iters >= 0:
            robot_rigid_kwargs["solver_position_iteration_count"] = args.robot_solver_position_iters
        if args.robot_solver_velocity_iters >= 0:
            robot_rigid_kwargs["solver_velocity_iteration_count"] = args.robot_solver_velocity_iters
        if args.robot_max_depenetration_velocity >= 0.0:
            robot_rigid_kwargs["max_depenetration_velocity"] = args.robot_max_depenetration_velocity
        actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
                                       stiffness=400.0, damping=80.0),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=gripper_joint_expr,
                stiffness=args.gripper_stiffness,
                damping=args.gripper_damping,
                effort_limit=args.gripper_effort_limit,
            ),
        }
        if (
            args.passive_gripper_stiffness > 0.0
            or args.passive_gripper_damping > 0.0
            or args.passive_gripper_effort_limit > 0.0
        ):
            actuators["gripper_passive_damping"] = ImplicitActuatorCfg(
                joint_names_expr=passive_gripper_joint_expr,
                stiffness=args.passive_gripper_stiffness,
                damping=args.passive_gripper_damping,
                effort_limit=args.passive_gripper_effort_limit,
            )
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/work/data/assets/ur3e_robotiq_vbts.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**robot_rigid_kwargs) if robot_rigid_kwargs else None,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, args.robot_z),
                rot=tuple(_quat_from_rpy_deg(args.robot_root_rpy_deg)),
            ),
            actuators=actuators,
        )
        robot = Articulation(robot_cfg)
        object_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.object_mu,
            dynamic_friction=args.object_mu,
            friction_combine_mode=args.friction_combine_mode,
            restitution_combine_mode="min",
            restitution=0.0,
        )
        rigid_kwargs = {}
        if args.object_solver_position_iters >= 0:
            rigid_kwargs["solver_position_iteration_count"] = args.object_solver_position_iters
        if args.object_solver_velocity_iters >= 0:
            rigid_kwargs["solver_velocity_iteration_count"] = args.object_solver_velocity_iters
        if args.object_max_depenetration_velocity >= 0.0:
            rigid_kwargs["max_depenetration_velocity"] = args.object_max_depenetration_velocity
        object_rigid_props = sim_utils.RigidBodyPropertiesCfg(**rigid_kwargs)
        object_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=args.object_contact_offset,
            rest_offset=args.object_rest_offset,
        )
        if args.object == "sphere":
            spawn = sim_utils.SphereCfg(radius=args.object_size / 2.0,
                                        rigid_props=object_rigid_props,
                                        mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass),
                                        collision_props=object_collision_props,
                                        physics_material=object_material)
        elif args.object == "cylinder":
            spawn = sim_utils.CylinderCfg(radius=args.object_size / 2.0, height=args.object_size,
                                          rigid_props=object_rigid_props,
                                          mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass),
                                          collision_props=object_collision_props,
                                          physics_material=object_material)
        else:
            spawn = sim_utils.CuboidCfg(size=(args.object_size, args.object_size, args.object_size),
                                        rigid_props=object_rigid_props,
                                        mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass),
                                        collision_props=object_collision_props,
                                        physics_material=object_material)
        obj = RigidObject(RigidObjectCfg(
            prim_path="/World/Object",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, args.object_size / 2.0 + 0.02)),
        ))
        manifest = json.loads(Path(manifest_path).read_text()) if Path(manifest_path).exists() else {}

        def scene_robot_path(path: str) -> str:
            if path.startswith("/Robot/"):
                return "/World/Robot/" + path[len("/Robot/"):]
            if path == "/Robot":
                return "/World/Robot"
            return path

        contact_sensors = {}
        contact_force_info = {
            "source": "none",
            "detail": "",
            "force_gate_N": float(args.physics_force_gate_N),
        }

        def setup_contact_sensors(kind: str) -> bool:
            nonlocal contact_sensors, contact_force_info
            try:
                from isaaclab.sensors import ContactSensor, ContactSensorCfg

                sensors = {}
                for pad in manifest.get("pads", []):
                    side = pad["name"]
                    if kind == "gel_collision":
                        # ContactSensor must bind to a rigid body prim. The gel collision
                        # mesh is owned by the pad link, so this tier keeps the gel source
                        # label while binding the reporter to the rigid pad body.
                        prim_path = scene_robot_path(pad["pad_link"])
                    elif kind == "pad_link":
                        prim_path = scene_robot_path(pad["pad_link"])
                    else:
                        raise ValueError(kind)
                    sensors[side] = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path,
                        update_period=0.0,
                        history_length=1,
                        debug_vis=False,
                        filter_prim_paths_expr=["/World/Object"],
                    ))
                if not sensors:
                    return False
                contact_sensors = sensors
                contact_force_info = {
                    "source": f"isaaclab.sensors.ContactSensor:{kind}",
                    "detail": "net_forces_w projected onto each live gel normal; filtered to /World/Object",
                    "force_gate_N": float(args.physics_force_gate_N),
                }
                return True
            except Exception as exc:
                contact_force_info = {
                    "source": f"ContactSensor:{kind}:failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "force_gate_N": float(args.physics_force_gate_N),
                }
                flog(f"CONTACT_SENSOR_SETUP_FAIL kind={kind} {type(exc).__name__}: {exc}")
                contact_sensors = {}
                return False

        if not setup_contact_sensors("gel_collision"):
            setup_contact_sensors("pad_link")
        jaw_l = jaw_r = jaw_shelf = pedestal = None
        pseudo_jaw_thickness = 0.006
        pseudo_shelf_thickness = 0.002
        if args.pseudo_jaw_z > 0.0:
            pseudo_jaw_center_z = args.pseudo_jaw_z
        elif args.pseudo_jaw_support_probe:
            pseudo_jaw_center_z = max(0.030, args.object_size / 2.0 + pseudo_shelf_thickness + 0.002)
        else:
            pseudo_jaw_center_z = args.object_size / 2.0
        pseudo_shelf_center_z = pseudo_jaw_center_z - args.object_size / 2.0 - pseudo_shelf_thickness / 2.0 + 1.0e-4
        pseudo_jaw_open = (
            args.pseudo_jaw_open_half_gap
            if args.pseudo_jaw_open_half_gap > 0.0
            else args.object_size / 2.0 + pseudo_jaw_thickness / 2.0 + 0.012
        )
        pseudo_jaw_close = (
            args.pseudo_jaw_close_half_gap
            if args.pseudo_jaw_close_half_gap > 0.0
            else max(
                pseudo_jaw_thickness / 2.0 + 0.0005,
                args.object_size / 2.0 + pseudo_jaw_thickness / 2.0 - args.pseudo_jaw_indent,
            )
        )
        pseudo_jaw_pitch_rad = np.deg2rad(args.pseudo_jaw_pitch_deg)
        R_jaw_l = _Rx(-pseudo_jaw_pitch_rad)
        R_jaw_r = _Rx(pseudo_jaw_pitch_rad)
        q_jaw_l = _quat_wxyz_from_R(R_jaw_l)
        q_jaw_r = _quat_wxyz_from_R(R_jaw_r)
        if args.pseudo_jaw_probe:
            jaw_height = max(args.object_size, 0.010)
            jaw_spawn = sim_utils.CuboidCfg(
                size=(max(0.028, args.object_size * 1.8), pseudo_jaw_thickness, jaw_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=object_material,
            )
            jaw_l = RigidObject(RigidObjectCfg(
                prim_path="/World/PseudoJawL",
                spawn=jaw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, pseudo_jaw_open, pseudo_jaw_center_z)),
            ))
            jaw_r = RigidObject(RigidObjectCfg(
                prim_path="/World/PseudoJawR",
                spawn=jaw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -pseudo_jaw_open, pseudo_jaw_center_z)),
            ))
            if args.pseudo_jaw_support_probe:
                shelf_spawn = sim_utils.CuboidCfg(
                    size=(max(0.026, args.object_size * 1.8), max(0.020, args.object_size * 1.4), pseudo_shelf_thickness),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=object_material,
                )
                jaw_shelf = RigidObject(RigidObjectCfg(
                    prim_path="/World/PseudoJawSupport",
                    spawn=shelf_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, pseudo_shelf_center_z)),
                ))
        if args.physics_pedestal_probe:
            pedestal_spawn = sim_utils.CuboidCfg(
                size=(args.physics_pedestal_size, args.physics_pedestal_size, args.physics_pedestal_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=args.object_contact_offset,
                    rest_offset=args.object_rest_offset,
                ),
                physics_material=object_material,
            )
            pedestal = RigidObject(RigidObjectCfg(
                prim_path="/World/GraspPedestal",
                spawn=pedestal_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, args.physics_pedestal_height / 2.0)),
            ))
        sim.reset()
        robot.update(1.0 / 120.0)
        obj.update(1.0 / 120.0)
        if jaw_l is not None:
            jaw_l.update(1.0 / 120.0)
            jaw_r.update(1.0 / 120.0)
            if jaw_shelf is not None:
                jaw_shelf.update(1.0 / 120.0)
        if pedestal is not None:
            pedestal.update(1.0 / 120.0)
        robot_root_hold_pose = None
        if args.physics_lift_probe and hasattr(robot, "write_root_pose_to_sim"):
            import torch

            robot_root_hold_pose = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=1).clone()

        def write_robot_root_hold():
            if robot_root_hold_pose is None:
                return
            import torch

            robot.write_root_pose_to_sim(robot_root_hold_pose)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))

        pedestal_hold = {"pos": None, "quat": (1.0, 0.0, 0.0, 0.0)}

        def write_pedestal_hold():
            if pedestal is None or pedestal_hold["pos"] is None:
                return
            import torch

            st = pedestal.data.default_root_state.clone()
            st[0, 0:3] = torch.tensor(pedestal_hold["pos"], device=pedestal.device, dtype=torch.float32)
            st[0, 3:7] = torch.tensor(pedestal_hold["quat"], device=pedestal.device, dtype=torch.float32)
            pedestal.write_root_pose_to_sim(st[:, :7])
            pedestal.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))

        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)
        arm_ids = [
            i for i, name in enumerate(joint_names)
            if name.startswith("shoulder") or name.startswith("elbow") or name.startswith("wrist")
        ]
        arm_hold_target = robot.data.joint_pos[0, arm_ids].detach().clone() if args.physics_lift_probe and arm_ids else None
        if args.gripper_control_mode == "all":
            gripper_ids = [
                i for i, name in enumerate(joint_names)
                if name == "finger_joint" or "finger" in name or "knuckle" in name
            ]
        else:
            gripper_ids = [i for i, name in enumerate(joint_names) if name == "finger_joint"]
        passive_log_names = [
            "left_outer_finger_joint",
            "right_outer_finger_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_log_ids = [
            ([i for i, name in enumerate(joint_names) if name == wanted] or [None])[0]
            for wanted in passive_log_names
        ]
        gripper_control = False
        if gripper_ids and hasattr(robot, "set_joint_position_target"):
            import torch

            def set_gripper(target_value, steps):
                target = torch.full((1, len(gripper_ids)), float(target_value), device=robot.device)
                arm_target = arm_hold_target.unsqueeze(0) if arm_hold_target is not None else None
                for _ in range(int(steps)):
                    if args.physics_lift_probe and arm_ids and arm_target is not None:
                        robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
                    robot.set_joint_position_target(target, joint_ids=gripper_ids)
                    robot.write_data_to_sim()
                    write_robot_root_hold()
                    write_pedestal_hold()
                    sim.step()
                    robot.update(1.0 / 120.0)
                    obj.update(1.0 / 120.0)
                    if pedestal is not None:
                        pedestal.update(1.0 / 120.0)

            if args.physics_lift_probe:
                set_gripper(args.gripper_open_target, 20)
            elif args.gripper_scan_probe:
                pass
            else:
                set_gripper(args.gripper_target, 20)
            gripper_control = True
        else:
            def set_gripper(_target_value, _steps):
                return None
        duplicate_body_names = {
            name: [i for i, candidate in enumerate(body_names) if candidate == name]
            for name in sorted(set(body_names))
            if body_names.count(name) > 1
        }
        pad_body_matches = {
            side: [i for i, name in enumerate(body_names) if name == f"{side}_inner_finger"]
            for side in ("left", "right")
        }
        pad_indices = {
            side: (matches[0] if matches else None)
            for side, matches in pad_body_matches.items()
        }

        def current_pad_pose_w():
            out = {}
            for side in ("left", "right"):
                idx = pad_indices.get(side)
                if idx is None:
                    continue
                out[side] = {
                    "pos": robot.data.body_pos_w[0, idx].detach().cpu().tolist(),
                    "quat_wxyz": robot.data.body_quat_w[0, idx].detach().cpu().tolist(),
                }
            return out

        def gel_gap_from_frames(frames):
            if "left" not in frames or "right" not in frames:
                return {"frames": frames, "midpoint": None, "gap_m": None}
            left = np.asarray(frames["left"]["origin"], dtype=np.float64)
            right = np.asarray(frames["right"]["origin"], dtype=np.float64)
            left_n = np.asarray(frames["left"]["normal"], dtype=np.float64)
            right_n = np.asarray(frames["right"]["normal"], dtype=np.float64)
            left_n = left_n / (np.linalg.norm(left_n) + 1.0e-12)
            right_n = right_n / (np.linalg.norm(right_n) + 1.0e-12)
            origin_delta = right - left
            left_face_gap = float(origin_delta @ left_n)
            right_face_gap = float((-origin_delta) @ right_n)
            face_gap = 0.5 * (left_face_gap + right_face_gap)
            origin_gap = float(np.linalg.norm(left - right))
            normal_dot = float(left_n @ right_n)
            return {
                "frames": frames,
                "midpoint": ((left + right) * 0.5).tolist(),
                "gap_m": float(face_gap) if np.isfinite(face_gap) else None,
                "origin_gap_m": origin_gap if np.isfinite(origin_gap) else None,
                "left_face_gap_m": left_face_gap if np.isfinite(left_face_gap) else None,
                "right_face_gap_m": right_face_gap if np.isfinite(right_face_gap) else None,
                "normal_dot": normal_dot if np.isfinite(normal_dot) else None,
            }

        def current_gel_origins_w(pad_pose=None):
            pad_pose = current_pad_pose_w() if pad_pose is None else pad_pose
            out = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                if side not in pad_pose:
                    continue
                T_w_link = pose_to_T(pad_pose[side]["pos"], pad_pose[side]["quat_wxyz"])
                T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
                T_w_gel = T_w_link @ T_link_gel
                out[side] = T_w_gel[:3, 3].tolist()
            return out

        def current_gel_frames_w(pad_pose=None):
            pad_pose = current_pad_pose_w() if pad_pose is None else pad_pose
            out = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                if side not in pad_pose:
                    continue
                T_w_link = pose_to_T(pad_pose[side]["pos"], pad_pose[side]["quat_wxyz"])
                T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
                T_w_gel = T_w_link @ T_link_gel
                out[side] = {
                    "origin": T_w_gel[:3, 3].tolist(),
                    "x": T_w_gel[:3, 0].tolist(),
                    "y": T_w_gel[:3, 1].tolist(),
                    "normal": T_w_gel[:3, 2].tolist(),
                }
            return out

        def gel_gap_state():
            frames = current_gel_frames_w()
            return gel_gap_from_frames(frames)

        def _T_from_gel_frame(frame):
            T = np.eye(4, dtype=np.float64)
            T[:3, 0] = np.asarray(frame["x"], dtype=np.float64)
            T[:3, 1] = np.asarray(frame["y"], dtype=np.float64)
            T[:3, 2] = np.asarray(frame["normal"], dtype=np.float64)
            T[:3, 3] = np.asarray(frame["origin"], dtype=np.float64)
            return T

        def object_center_from_gel_frames(frames, contact_depth):
            origins = []
            normals = []
            desired_offsets = []
            radius = args.object_size / 2.0
            desired = radius - float(contact_depth)
            for side in ("left", "right"):
                frame = frames.get(side)
                if frame is None:
                    continue
                T_w_gel = _T_from_gel_frame(frame)
                origins.append(T_w_gel[:3, 3].copy())
                normals.append(T_w_gel[:3, 2].copy())
                desired_offsets.append(desired)
            if len(origins) >= 2:
                A = np.stack([n / (np.linalg.norm(n) + 1.0e-12) for n in normals], axis=0)
                b = np.asarray([
                    float(n @ o + d)
                    for n, o, d in zip(A, origins, desired_offsets)
                ], dtype=np.float64)
                p0 = np.mean(np.stack(origins, axis=0), axis=0)
                gram = A @ A.T
                try:
                    correction = A.T @ np.linalg.solve(gram + 1.0e-10 * np.eye(gram.shape[0]), A @ p0 - b)
                    return p0 - correction
                except np.linalg.LinAlgError:
                    pass
            if origins:
                centers = []
                for origin, normal in zip(origins, normals):
                    centers.append(origin + normal / (np.linalg.norm(normal) + 1.0e-12) * desired)
                return np.mean(np.stack(centers, axis=0), axis=0)
            return np.array([0.45, 0.0, 0.02], dtype=np.float64)

        def predicted_contact_for_gel_frames(center_np, frames, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
            T_w_obj = pose_to_T(np.asarray(center_np, dtype=np.float64).tolist(), quat_wxyz)
            pen = {}
            for side, frame in frames.items():
                T = T_inv(_T_from_gel_frame(frame)) @ T_w_obj
                pen[side] = float(_pen_for_object(args.object, args.object_size, T, gel_grid(32, 0.009)).max())
            return pen

        def current_live_contact_snapshot():
            pad_pose = current_pad_pose_w()
            obj_pose = {
                "pos": obj.data.root_pos_w[0].detach().cpu().tolist(),
                "quat_wxyz": obj.data.root_quat_w[0].detach().cpu().tolist(),
            }
            T_w_obj = pose_to_T(obj_pose["pos"], obj_pose["quat_wxyz"])
            pen = {}
            T_gel_obj = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                if side not in pad_pose:
                    continue
                T_w_link = pose_to_T(pad_pose[side]["pos"], pad_pose[side]["quat_wxyz"])
                T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
                T = T_inv(T_w_link @ T_link_gel) @ T_w_obj
                T_gel_obj[side] = T.tolist()
                pen[side] = float(_pen_for_object(args.object, args.object_size, T, gel_grid(32, 0.009)).max())
            return {
                "object_pose_w": obj_pose,
                "pad_pose_w": pad_pose,
                "gel_origin_w": current_gel_origins_w(pad_pose),
                "T_gel_obj": T_gel_obj,
                "pen_m": pen,
            }

        def _to_numpy(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float64)

        def _first_force_vec(arr):
            arr = _to_numpy(arr)
            while arr.ndim > 1:
                arr = arr[0]
            if arr.shape[0] != 3:
                return np.zeros(3, dtype=np.float64)
            return arr.astype(np.float64)

        def _select_body_force(arr, body_idx):
            arr = _to_numpy(arr)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2 and body_idx is not None and body_idx < arr.shape[0]:
                return arr[body_idx].astype(np.float64)
            if arr.ndim == 1 and arr.shape[0] == 3:
                return arr.astype(np.float64)
            return np.zeros(3, dtype=np.float64)

        def current_contact_forces():
            nonlocal contact_force_info
            frames = current_gel_frames_w()
            forces = {}
            source = contact_force_info.get("source", "none")
            if contact_sensors:
                try:
                    for sensor in contact_sensors.values():
                        sensor.update(1.0 / 120.0)
                    for side, sensor in contact_sensors.items():
                        vec = _first_force_vec(sensor.data.net_forces_w)
                        normal = np.asarray(frames.get(side, {}).get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
                        normal = normal / (np.linalg.norm(normal) + 1.0e-12)
                        forces[side] = {
                            "world_N": vec.tolist(),
                            "normal_N": float(abs(vec @ normal)),
                        }
                    return {"source": source, "forces": forces}
                except Exception as exc:
                    flog(f"CONTACT_SENSOR_READ_FAIL source={source} {type(exc).__name__}: {exc}")
                    if "gel_collision" in source and setup_contact_sensors("pad_link"):
                        return current_contact_forces()
                    contact_force_info = {
                        "source": "robot.root_physx_view.get_net_contact_forces",
                        "detail": f"ContactSensor read failed: {type(exc).__name__}: {exc}",
                        "force_gate_N": float(args.physics_force_gate_N),
                    }
            try:
                view = robot.root_physx_view
                raw = None
                last_exc = None
                for name in ("get_net_contact_forces", "get_contact_forces"):
                    fn = getattr(view, name, None)
                    if fn is None:
                        continue
                    for call_args in ((), (1.0 / 120.0,)):
                        try:
                            raw = fn(*call_args)
                            contact_force_info["source"] = f"robot.root_physx_view.{name}"
                            contact_force_info["detail"] = "fallback pad-link net force projected onto live gel normals"
                            break
                        except Exception as exc:
                            last_exc = exc
                    if raw is not None:
                        break
                if raw is None:
                    raise RuntimeError("no robot contact-force API succeeded" if last_exc is None else str(last_exc))
                for side, idx in pad_indices.items():
                    vec = _select_body_force(raw, idx)
                    normal = np.asarray(frames.get(side, {}).get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
                    normal = normal / (np.linalg.norm(normal) + 1.0e-12)
                    forces[side] = {
                        "world_N": vec.tolist(),
                        "normal_N": float(abs(vec @ normal)),
                    }
                return {"source": contact_force_info.get("source", "robot.root_physx_view"), "forces": forces}
            except Exception as exc:
                contact_force_info = {
                    "source": "none",
                    "detail": f"no usable contact-force source: {type(exc).__name__}: {exc}",
                    "force_gate_N": float(args.physics_force_gate_N),
                }
                return {"source": "none", "forces": {}}

        def contact_gate_pass(force_snapshot):
            forces = force_snapshot.get("forces", {})
            vals = [
                float(forces.get(side, {}).get("normal_N", 0.0))
                for side in ("left", "right")
            ]
            return bool(len(vals) == 2 and min(vals) >= args.physics_force_gate_N)

        def current_passive_joint_pos():
            joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
            out = []
            for idx in passive_log_ids:
                out.append(float(joint_pos[idx]) if idx is not None else float("nan"))
            return out

        def physics_trace_row(phase, target_value=None):
            snap = current_live_contact_snapshot()
            force = current_contact_forces()
            joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
            obj_pos = obj.data.root_pos_w[0].detach().cpu().tolist()
            root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            return {
                "phase": phase,
                "target": None if target_value is None else float(target_value),
                "finger_joint_pos": float(joint_pos[gripper_ids[0]]) if gripper_ids else None,
                "passive_joint_names": passive_log_names,
                "passive_joint_pos": current_passive_joint_pos(),
                "pen_m": snap["pen_m"],
                "force_N": {
                    side: float(force.get("forces", {}).get(side, {}).get("normal_N", 0.0))
                    for side in ("left", "right")
                },
                "force_world_N": {
                    side: force.get("forces", {}).get(side, {}).get("world_N", [0.0, 0.0, 0.0])
                    for side in ("left", "right")
                },
                "force_source": force.get("source", "none"),
                "object_pos_w": obj_pos,
                "robot_root_z_m": float(root_pos[2]),
            }

        def assert_placement_within_recess(label):
            if not args.object_at_pads:
                return None
            snap = current_live_contact_snapshot()
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            limit = min(recess, args.max_live_placement_pen)
            bad = {side: pen for side, pen in snap["pen_m"].items() if pen >= limit}
            if bad:
                raise RuntimeError(
                    f"{label} placement exceeds gel recess guard: pen_m={bad}, "
                    f"limit={limit:.6g}, recess={recess:.6g}. "
                    "Use a smaller --live-contact-depth or open/slow-close instead of teleporting into a closed gap."
                )
            return snap

        def ramp_gripper_until_contact(start_target, end_target, steps):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return {"enabled": False, "rows": []}
            import torch

            rows = []
            prev_finger = None
            stall_count = 0
            stopped_reason = "completed"
            final_target = float(start_target)
            for k in range(int(max(steps, 1))):
                alpha = (k + 1) / float(max(steps, 1))
                target_value = (1.0 - alpha) * float(start_target) + alpha * float(end_target)
                target = torch.full((1, len(gripper_ids)), target_value, device=robot.device)
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.set_joint_position_target(target, joint_ids=gripper_ids)
                robot.write_data_to_sim()
                write_robot_root_hold()
                write_pedestal_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)
                snap = current_live_contact_snapshot()
                pen_vals = list(snap["pen_m"].values())
                min_pen = min(pen_vals) if pen_vals else 0.0
                max_pen = max(pen_vals) if pen_vals else 0.0
                joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                finger_pos = float(joint_pos[gripper_ids[0]]) if gripper_ids else None
                if prev_finger is not None and finger_pos is not None:
                    if abs(finger_pos - prev_finger) < args.physics_close_stall_eps:
                        stall_count += 1
                    else:
                        stall_count = 0
                prev_finger = finger_pos
                if k == 0 or (k + 1) % max(args.physics_close_log_every, 1) == 0:
                    force = current_contact_forces()
                    rows.append({
                        "step": int(k + 1),
                        "target": float(target_value),
                        "finger_joint_pos": finger_pos,
                        "min_pen_m": float(min_pen),
                        "max_pen_m": float(max_pen),
                        "force_N": {
                            side: float(force.get("forces", {}).get(side, {}).get("normal_N", 0.0))
                            for side in ("left", "right")
                        },
                        "force_source": force.get("source", "none"),
                    })
                final_target = float(target_value)
                if pen_vals and min_pen >= args.physics_close_stop_pen:
                    stopped_reason = "pen_threshold"
                    break
                if stall_count >= args.physics_close_stall_steps:
                    stopped_reason = "joint_stall"
                    break
            return {
                "enabled": True,
                "steps_requested": int(steps),
                "steps_run": int(k + 1),
                "start_target": float(start_target),
                "end_target": float(end_target),
                "final_target": float(final_target),
                "stop_pen_m": float(args.physics_close_stop_pen),
                "stall_eps": float(args.physics_close_stall_eps),
                "stall_steps": int(args.physics_close_stall_steps),
                "stopped_reason": stopped_reason,
                "rows": rows,
                "final_snapshot": current_live_contact_snapshot(),
            }

        def replay_gripper_ramp(start_target, end_target, steps):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return
            import torch

            for k in range(int(max(steps, 1))):
                alpha = (k + 1) / float(max(steps, 1))
                target_value = (1.0 - alpha) * float(start_target) + alpha * float(end_target)
                target = torch.full((1, len(gripper_ids)), target_value, device=robot.device)
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.set_joint_position_target(target, joint_ids=gripper_ids)
                robot.write_data_to_sim()
                write_robot_root_hold()
                write_pedestal_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)

        def squeezed_target_from(final_target, end_target):
            direction = 1.0 if float(end_target) >= float(final_target) else -1.0
            squeezed = float(final_target) + direction * float(args.physics_squeeze_margin)
            if direction > 0.0:
                return min(float(end_target), squeezed)
            return max(float(end_target), squeezed)

        def run_gripper_target_steps(target_value, steps, phase, trace_rows, root_pose_fn=None):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return []
            import torch

            phase_rows = []
            target = torch.full((1, len(gripper_ids)), float(target_value), device=robot.device)
            for k in range(int(max(steps, 0))):
                if root_pose_fn is not None:
                    root_pose, root_vel = root_pose_fn(k)
                    robot.write_root_pose_to_sim(root_pose)
                    if hasattr(robot, "write_root_velocity_to_sim"):
                        robot.write_root_velocity_to_sim(root_vel)
                elif args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.set_joint_position_target(target, joint_ids=gripper_ids)
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.write_data_to_sim()
                if root_pose_fn is None:
                    write_robot_root_hold()
                write_pedestal_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)
                row = physics_trace_row(phase, target_value)
                row["step_in_phase"] = int(k + 1)
                trace_rows.append(row)
                phase_rows.append(row)
            return phase_rows

        def calibrate_close_ramp(start_target, end_target, steps):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return {"enabled": False, "rows": []}
            import torch

            rows = []
            target_gap = float(args.object_size + 2.0 * args.physics_close_stop_pen)
            preclose_gap = float(args.object_size + args.physics_preclose_clearance)
            for k in range(int(max(steps, 1))):
                alpha = (k + 1) / float(max(steps, 1))
                target_value = (1.0 - alpha) * float(start_target) + alpha * float(end_target)
                target = torch.full((1, len(gripper_ids)), target_value, device=robot.device)
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.set_joint_position_target(target, joint_ids=gripper_ids)
                robot.write_data_to_sim()
                write_robot_root_hold()
                write_pedestal_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)
                state = gel_gap_state()
                joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                row = {
                    "step": int(k + 1),
                    "target": float(target_value),
                    "finger_target": float(target_value),
                    "finger_joint_pos": float(joint_pos[gripper_ids[0]]) if gripper_ids else None,
                    "gap_m": state["gap_m"],
                    "origin_gap_m": state.get("origin_gap_m"),
                    "left_face_gap_m": state.get("left_face_gap_m"),
                    "right_face_gap_m": state.get("right_face_gap_m"),
                    "normal_dot": state.get("normal_dot"),
                    "midpoint": state["midpoint"],
                    "frames": state["frames"],
                    "target_gap_error_m": None if state["gap_m"] is None else float(abs(state["gap_m"] - target_gap)),
                    "preclose_gap_error_m": None if state["gap_m"] is None else float(abs(state["gap_m"] - preclose_gap)),
                }
                rows.append(row)
                if k == 0 or (k + 1) % max(args.physics_close_log_every, 1) == 0 or k + 1 == int(max(steps, 1)):
                    flog(
                        "calibrated_close_row "
                        f"step={row['step']} target={row['target']:.6g} "
                        f"finger={row['finger_joint_pos']} gap={row['gap_m']} "
                        f"origin_gap={row['origin_gap_m']} "
                        f"face_L={row['left_face_gap_m']} face_R={row['right_face_gap_m']} "
                        f"normal_dot={row['normal_dot']}"
                    )
            valid = [r for r in rows if r["gap_m"] is not None and r["midpoint"] is not None]
            preclose_row = min(valid, key=lambda r: r["preclose_gap_error_m"]) if valid else None
            place_center = None
            place_row = None
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            placement_recess_margin = 5.0e-5
            placement_cap = min(
                max(float(args.live_contact_depth), 0.0),
                max(recess - placement_recess_margin, 0.0),
                max(float(args.max_live_placement_pen), 0.0),
            )
            placement_target_pen = min(max(float(args.place_target_pen), 0.0), placement_cap)
            desired_gap = max(float(args.object_size) - 2.0 * placement_target_pen, 0.0)
            for row in valid:
                row["placement_cap_m"] = float(placement_cap)
                row["placement_recess_margin_m"] = float(placement_recess_margin)
                row["placement_target_pen_m"] = float(placement_target_pen)
                row["placement_desired_gap_m"] = float(desired_gap)
                row["placement_desired_gap_error_m"] = float(abs(row["gap_m"] - desired_gap))
            bracket = None
            if placement_target_pen > 0.0:
                if valid and abs(float(valid[0]["gap_m"]) - desired_gap) <= 1.0e-12:
                    bracket = (valid[0], valid[0])
                for prev, curr in zip(valid, valid[1:]):
                    if bracket is not None:
                        break
                    prev_gap = float(prev["gap_m"])
                    curr_gap = float(curr["gap_m"])
                    prev_delta = prev_gap - desired_gap
                    curr_delta = curr_gap - desired_gap
                    if abs(curr_delta) <= 1.0e-12 or prev_delta * curr_delta < 0.0:
                        bracket = (prev, curr)
                        break
            if bracket is not None:
                prev, curr = bracket
                prev_gap = float(prev["gap_m"])
                curr_gap = float(curr["gap_m"])
                denom = curr_gap - prev_gap
                frac = 0.0 if abs(denom) <= 1.0e-12 else (desired_gap - prev_gap) / denom
                frac = float(min(1.0, max(0.0, frac)))
                interp_target = (1.0 - frac) * float(prev["target"]) + frac * float(curr["target"])
                interp_step = (1.0 - frac) * float(prev["step"]) + frac * float(curr["step"])
                place_row = {
                    "step": float(interp_step),
                    "target": float(interp_target),
                    "finger_target": float(interp_target),
                    "finger_joint_pos": None,
                    "gap_m": float(desired_gap),
                    "midpoint": None,
                    "frames": None,
                    "target_gap_error_m": float(abs(desired_gap - target_gap)),
                    "preclose_gap_error_m": float(abs(desired_gap - preclose_gap)),
                    "place_depth_mode": "interpolated_live",
                    "placement_depth_m": float(placement_target_pen),
                    "placement_target_pen_m": float(placement_target_pen),
                    "placement_desired_gap_m": float(desired_gap),
                    "placement_cap_m": float(placement_cap),
                    "placement_recess_margin_m": float(placement_recess_margin),
                    "bracket_fraction": float(frac),
                    "bracket_rows": [
                        {
                            "step": int(prev["step"]),
                            "target": float(prev["target"]),
                            "finger_target": float(prev["target"]),
                            "gap_m": float(prev_gap),
                            "origin_gap_m": prev.get("origin_gap_m"),
                            "left_face_gap_m": prev.get("left_face_gap_m"),
                            "right_face_gap_m": prev.get("right_face_gap_m"),
                        },
                        {
                            "step": int(curr["step"]),
                            "target": float(curr["target"]),
                            "finger_target": float(curr["target"]),
                            "gap_m": float(curr_gap),
                            "origin_gap_m": curr.get("origin_gap_m"),
                            "left_face_gap_m": curr.get("left_face_gap_m"),
                            "right_face_gap_m": curr.get("right_face_gap_m"),
                        },
                    ],
                }
                preclose_row = place_row
            if place_row is not None:
                place_row["place_center"] = None
                place_row["predicted_pen_m"] = None
                preclose_row = place_row
            gap_values = [float(r["gap_m"]) for r in valid]
            origin_gap_values = [
                float(r["origin_gap_m"])
                for r in valid
                if r.get("origin_gap_m") is not None
            ]
            failure_detail = None
            if place_row is None:
                if gap_values:
                    failure_detail = (
                        "no calibrated close row bracket crossed desired placement gap "
                        f"{desired_gap:.6g} m for target_pen={placement_target_pen:.6g} m "
                        f"within cap={placement_cap:.6g} m; "
                        f"rows={len(rows)} valid_rows={len(valid)} "
                        f"first_gap={gap_values[0]:.6g} m last_gap={gap_values[-1]:.6g} m "
                        f"min_gap={min(gap_values):.6g} m max_gap={max(gap_values):.6g} m"
                    )
                    if origin_gap_values:
                        failure_detail += (
                            f" first_origin_gap={origin_gap_values[0]:.6g} m "
                            f"last_origin_gap={origin_gap_values[-1]:.6g} m"
                        )
                else:
                    failure_detail = (
                        "no calibrated close row bracket crossed desired placement gap "
                        f"{desired_gap:.6g} m for target_pen={placement_target_pen:.6g} m "
                        f"within cap={placement_cap:.6g} m; rows={len(rows)} valid_rows=0"
                    )
            return {
                "enabled": True,
                "target_gap_m": target_gap,
                "preclose_gap_m": preclose_gap,
                "placement_cap_m": float(placement_cap),
                "placement_recess_margin_m": float(placement_recess_margin),
                "placement_target_pen_m": float(placement_target_pen),
                "placement_desired_gap_m": float(desired_gap),
                "place_center": None if place_center is None else place_center.tolist(),
                "place_row": place_row,
                "preclose_row": preclose_row,
                "gap_metric": "mean signed distance between gel face planes along inward pad normals",
                "origin_gap_metric": "Euclidean distance between gel frame origins",
                "placement_failure": failure_detail,
                "rows": rows,
            }

        def pair_distance(points):
            if "left" not in points or "right" not in points:
                return None
            return float(np.linalg.norm(np.asarray(points["left"]) - np.asarray(points["right"])))

        def _stage_xform_frame(stage, path: str):
            try:
                from pxr import Gf, Usd, UsdGeom

                prim = stage.GetPrimAtPath(path)
                if prim is None or not prim.IsValid():
                    return {"path": path, "valid": False, "error": "prim not found"}
                mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
                origin = mat.ExtractTranslation()
                axes = []
                for vec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                    axis = mat.TransformDir(Gf.Vec3d(*vec))
                    axis_np = np.asarray([float(axis[0]), float(axis[1]), float(axis[2])], dtype=np.float64)
                    axes.append((axis_np / (np.linalg.norm(axis_np) + 1.0e-12)).tolist())
                return {
                    "path": path,
                    "valid": True,
                    "matrix": [[float(mat[i][j]) for j in range(4)] for i in range(4)],
                    "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
                    "x": axes[0],
                    "y": axes[1],
                    "normal": axes[2],
                }
            except Exception as exc:
                return {"path": path, "valid": False, "error": f"{type(exc).__name__}: {exc}"}

        def current_stage_gel_frames_w():
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            except Exception as exc:
                return {"_error": f"{type(exc).__name__}: {exc}"}
            out = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                frame = _stage_xform_frame(stage, scene_robot_path(pad["gel_frame_prim"]))
                if frame.get("valid"):
                    out[side] = {
                        "origin": frame["origin"],
                        "x": frame["x"],
                        "y": frame["y"],
                        "normal": frame["normal"],
                        "raw": frame,
                    }
                else:
                    out[side] = frame
            return out

        def current_stage_pad_link_frames_w():
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            except Exception as exc:
                return {"_error": f"{type(exc).__name__}: {exc}"}
            out = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                out[side] = _stage_xform_frame(stage, scene_robot_path(pad["pad_link"]))
            return out

        def frame_reconstruction_delta(reconstructed, stage_frames):
            out = {}
            for side in ("left", "right"):
                rec = reconstructed.get(side)
                raw = stage_frames.get(side)
                if rec is None or raw is None or not raw.get("origin"):
                    continue
                row = {}
                for axis in ("origin", "x", "y", "normal"):
                    a = np.asarray(rec[axis], dtype=np.float64)
                    b = np.asarray(raw[axis], dtype=np.float64)
                    if axis == "origin":
                        row["origin_delta_m"] = float(np.linalg.norm(a - b))
                    else:
                        a = a / (np.linalg.norm(a) + 1.0e-12)
                        b = b / (np.linalg.norm(b) + 1.0e-12)
                        row[f"{axis}_dot"] = float(a @ b)
                out[side] = row
            return out

        def run_gap_audit():
            rows = []
            targets = [float(x) for x in args.gap_audit_targets.split(",") if x.strip()]
            for target_value in targets:
                set_gripper(target_value, args.gap_audit_settle_steps)
                pad_pose = current_pad_pose_w()
                pad_points = {side: pose["pos"] for side, pose in pad_pose.items()}
                reconstructed = current_gel_frames_w(pad_pose)
                stage_gels = current_stage_gel_frames_w()
                valid_stage_gels = {
                    side: frame
                    for side, frame in stage_gels.items()
                    if isinstance(frame, dict) and frame.get("origin") is not None
                }
                stage_pad_links = current_stage_pad_link_frames_w()
                stage_pad_points = {
                    side: frame["origin"]
                    for side, frame in stage_pad_links.items()
                    if isinstance(frame, dict) and frame.get("origin") is not None
                }
                joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                rows.append({
                    "target": float(target_value),
                    "joint_pos": joint_pos,
                    "finger_joint_pos": float(joint_pos[gripper_ids[0]]) if gripper_ids else None,
                    "pad_body_pose_w": pad_pose,
                    "pad_body_pair_distance_m": pair_distance(pad_points),
                    "gel_reconstructed_w": reconstructed,
                    "gel_reconstructed_metrics": gel_gap_from_frames(reconstructed),
                    "gel_stage_w": stage_gels,
                    "gel_stage_metrics": gel_gap_from_frames(valid_stage_gels),
                    "pad_link_stage_w": stage_pad_links,
                    "pad_link_stage_pair_distance_m": pair_distance(stage_pad_points),
                    "reconstruction_vs_stage": frame_reconstruction_delta(reconstructed, valid_stage_gels),
                })
            return {
                "enabled": True,
                "targets": targets,
                "settle_steps": int(args.gap_audit_settle_steps),
                "body_name_duplicates": duplicate_body_names,
                "pad_body_name_matches": pad_body_matches,
                "pad_indices": pad_indices,
                "pad_index_note": (
                    "pad bodies are resolved by exact runtime body name; duplicated base_link names are recorded "
                    "but are not used for left/right_inner_finger lookup"
                ),
                "manifest_T_link_gel": {
                    pad["name"]: pad.get("T_link_gel")
                    for pad in manifest.get("pads", [])
                },
                "rows": rows,
            }

        gap_audit = run_gap_audit() if args.gap_audit else {"enabled": False}

        gripper_scan = {"enabled": False}
        if args.gripper_scan_probe:
            gripper_scan = {"enabled": True, "rows": []}
            targets = [float(x) for x in args.gripper_scan_targets.split(",") if x.strip()]
            for target_value in targets:
                set_gripper(target_value, args.gripper_scan_steps)
                pad_pose = current_pad_pose_w()
                pad_points = {side: pose["pos"] for side, pose in pad_pose.items()}
                gel_points = current_gel_origins_w(pad_pose)
                joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                joint_arr = np.asarray(joint_pos, dtype=np.float64)
                max_abs_joint_pos = float(np.nanmax(np.abs(joint_arr))) if joint_arr.size else None
                finite = bool(np.all(np.isfinite(joint_arr)))
                stable = bool(finite and max_abs_joint_pos is not None and max_abs_joint_pos < args.gripper_scan_stable_joint_abs)
                gripper_scan["rows"].append({
                    "target": float(target_value),
                    "joint_pos": joint_pos,
                    "finger_joint_pos": float(joint_pos[gripper_ids[0]]) if gripper_ids else None,
                    "max_abs_joint_pos": max_abs_joint_pos,
                    "pad_link_distance_m": pair_distance(pad_points),
                    "gel_origin_distance_m": pair_distance(gel_points),
                    "pad_pose_w": pad_pose,
                    "gel_origin_w": gel_points,
                    "finite": finite,
                    "stable": stable,
                })
                if not stable:
                    break

        def object_center_from_live_pads(contact_depth=None):
            centers = []
            origins = []
            normals = []
            desired_offsets = []
            frame_rows = []
            frame_targets = []
            frame_weights = []
            has_tangent_constraint = False
            depth = args.live_contact_depth if contact_depth is None else float(contact_depth)
            for pad in manifest.get("pads", []):
                side = pad["name"]
                idx = pad_indices.get(side)
                if idx is None:
                    continue
                T_w_link = pose_to_T(
                    robot.data.body_pos_w[0, idx].detach().cpu().tolist(),
                    robot.data.body_quat_w[0, idx].detach().cpu().tolist(),
                )
                T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
                T_w_gel = T_w_link @ T_link_gel
                radius = args.object_size / 2.0
                desired = radius - depth
                centers.append(T_w_gel @ np.array([0.0, 0.0, desired, 1.0]))
                origins.append(T_w_gel[:3, 3].copy())
                normals.append(T_w_gel[:3, 2].copy())
                desired_offsets.append(desired)
                tx_w = args.object_center_tangent_weight if args.object_center_tangent_x_weight < 0.0 else args.object_center_tangent_x_weight
                ty_w = args.object_center_tangent_weight if args.object_center_tangent_y_weight < 0.0 else args.object_center_tangent_y_weight
                for axis_idx, target, weight in (
                    (0, 0.0, tx_w),
                    (1, 0.0, ty_w),
                    (2, desired, 1.0),
                ):
                    if weight <= 0.0:
                        continue
                    if axis_idx in (0, 1):
                        has_tangent_constraint = True
                    axis = T_w_gel[:3, axis_idx].copy()
                    frame_rows.append(axis / (np.linalg.norm(axis) + 1e-12) * float(weight))
                    frame_targets.append(float(axis @ T_w_gel[:3, 3] + target) * float(weight))
                    frame_weights.append(float(weight))
            if args.object_center_solver == "lsq" and frame_rows and has_tangent_constraint:
                A_full = np.stack(frame_rows, axis=0)
                b_full = np.asarray(frame_targets, dtype=np.float64)
                try:
                    p, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
                    return p
                except np.linalg.LinAlgError:
                    flog("object_center frame-lsq failed; falling back to normal constraints")
            if args.object_center_solver == "lsq" and len(origins) >= 2:
                A = np.stack([n / (np.linalg.norm(n) + 1e-12) for n in normals], axis=0)
                b = np.asarray([
                    float(n @ o + d)
                    for n, o, d in zip(A, origins, desired_offsets)
                ], dtype=np.float64)
                p0 = np.mean(np.stack(origins, axis=0), axis=0)
                gram = A @ A.T
                try:
                    correction = A.T @ np.linalg.solve(gram + 1.0e-10 * np.eye(gram.shape[0]), A @ p0 - b)
                    return p0 - correction
                except np.linalg.LinAlgError:
                    flog("object_center_lsq failed; falling back to average center")
            return np.mean(np.stack(centers)[:, :3], axis=0) if centers else np.array([0.45, 0.0, 0.02])

        def place_object_at_live_pads(center_np=None):
            import torch

            center_np = object_center_from_live_pads() if center_np is None else np.asarray(center_np, dtype=np.float64)
            center_np = center_np + _vec3_arg(args.object_center_offset_world)
            center = torch.tensor(center_np, device=robot.device, dtype=torch.float32)
            pose = obj.data.default_root_state.clone()
            pose[0, 0:3] = center + torch.tensor([0.0, 0.0, args.object_pad_z_offset], device=robot.device)
            pose[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
            obj.write_root_pose_to_sim(pose[:, :7])
            obj.write_root_velocity_to_sim(torch.zeros_like(pose[:, 7:]))
            obj.update(1.0 / 120.0)

        def place_object_world(center_np, quat=(1.0, 0.0, 0.0, 0.0)):
            import torch

            pose = obj.data.default_root_state.clone()
            pose[0, 0:3] = torch.tensor(center_np, device=robot.device, dtype=torch.float32)
            pose[0, 3:7] = torch.tensor(quat, device=robot.device, dtype=torch.float32)
            obj.write_root_pose_to_sim(pose[:, :7])
            obj.write_root_velocity_to_sim(torch.zeros_like(pose[:, 7:]))
            obj.update(1.0 / 120.0)

        def place_pedestal_under(center_np):
            if pedestal is None:
                return None
            import torch

            center_np = np.asarray(center_np, dtype=np.float64)
            bottom_off = args.object_size / 2.0
            z = center_np[2] - bottom_off - args.physics_pedestal_height / 2.0
            st = pedestal.data.default_root_state.clone()
            st[0, 0:3] = torch.tensor([center_np[0], center_np[1], z], device=robot.device, dtype=torch.float32)
            st[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
            pedestal_hold["pos"] = [float(center_np[0]), float(center_np[1]), float(z)]
            pedestal_hold["quat"] = (1.0, 0.0, 0.0, 0.0)
            pedestal.write_root_pose_to_sim(st[:, :7])
            pedestal.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))
            pedestal.update(1.0 / 120.0)
            return {
                "center": [float(center_np[0]), float(center_np[1]), float(z)],
                "top_z": float(z + args.physics_pedestal_height / 2.0),
                "height": float(args.physics_pedestal_height),
                "size": float(args.physics_pedestal_size),
            }

        pseudo_probe = {"enabled": False}
        pseudo_T_gel_obj = {}
        pseudo_pen = {}
        if args.pseudo_jaw_probe and jaw_l is not None:
            import torch

            pseudo_probe = {
                "enabled": True,
                "jaw_center_z0": float(pseudo_jaw_center_z),
                "jaw_open_half_gap": float(pseudo_jaw_open),
                "jaw_close_half_gap": float(pseudo_jaw_close),
                "jaw_indent_m": float(args.pseudo_jaw_indent),
                "jaw_thickness_m": float(pseudo_jaw_thickness),
                "jaw_pitch_deg": float(args.pseudo_jaw_pitch_deg),
                "support_probe": bool(args.pseudo_jaw_support_probe),
                "support_thickness_m": float(pseudo_shelf_thickness) if args.pseudo_jaw_support_probe else 0.0,
            }
            center = torch.tensor([0.45, 0.0, pseudo_jaw_center_z], device=robot.device)
            pose = obj.data.default_root_state.clone()
            pose[0, 0:3] = center
            pose[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
            obj.write_root_pose_to_sim(pose[:, :7])
            obj.write_root_velocity_to_sim(torch.zeros_like(pose[:, 7:]))
            obj.update(1.0 / 120.0)

            def set_jaw(jaw, pos, vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
                st = jaw.data.default_root_state.clone()
                st[0, 0:3] = torch.tensor(pos, device=robot.device)
                st[0, 3:7] = torch.tensor(quat, device=robot.device)
                jaw.write_root_pose_to_sim(st[:, :7])
                root_vel = torch.zeros_like(st[:, 7:])
                root_vel[0, 0:3] = torch.tensor(vel, device=robot.device)
                jaw.write_root_velocity_to_sim(root_vel)

            set_jaw(jaw_l, [0.45, pseudo_jaw_open, pseudo_jaw_center_z], quat=q_jaw_l)
            set_jaw(jaw_r, [0.45, -pseudo_jaw_open, pseudo_jaw_center_z], quat=q_jaw_r)
            if jaw_shelf is not None:
                set_jaw(jaw_shelf, [0.45, 0.0, pseudo_shelf_center_z])
            preclose_settle_steps = (
                args.pseudo_jaw_preclose_settle_steps
                if args.pseudo_jaw_preclose_settle_steps >= 0
                else (8 if jaw_shelf is not None else 0)
            )
            pseudo_probe["preclose_settle_steps"] = int(preclose_settle_steps)
            for _ in range(int(preclose_settle_steps)):
                sim.step()
                obj.update(1.0 / 120.0)
                jaw_l.update(1.0 / 120.0)
                jaw_r.update(1.0 / 120.0)
                if jaw_shelf is not None:
                    jaw_shelf.update(1.0 / 120.0)

            y_open = pseudo_jaw_open
            y_close = pseudo_jaw_close
            dt = 1.0 / 120.0
            prev_y = y_open
            for k in range(args.pseudo_jaw_close_steps):
                a = (k + 1) / max(args.pseudo_jaw_close_steps, 1)
                y = (1.0 - a) * y_open + a * y_close
                vy = (y - prev_y) / dt
                set_jaw(jaw_l, [0.45, y, pseudo_jaw_center_z], [0.0, vy, 0.0], q_jaw_l)
                set_jaw(jaw_r, [0.45, -y, pseudo_jaw_center_z], [0.0, -vy, 0.0], q_jaw_r)
                sim.step()
                obj.update(1.0 / 120.0)
                jaw_l.update(1.0 / 120.0)
                jaw_r.update(1.0 / 120.0)
                if jaw_shelf is not None:
                    jaw_shelf.update(1.0 / 120.0)
                prev_y = y
            before = obj.data.root_pos_w[0].detach().cpu().tolist()
            pseudo_probe["object_pos_before_lift"] = before
            prev_z = pseudo_jaw_center_z
            prev_shelf_z = pseudo_shelf_center_z
            for k in range(args.pseudo_jaw_lift_steps):
                a = (k + 1) / max(args.pseudo_jaw_lift_steps, 1)
                z = pseudo_jaw_center_z + args.pseudo_jaw_lift_height * a
                vz = (z - prev_z) / dt
                set_jaw(jaw_l, [0.45, y_close, z], [0.0, 0.0, vz], q_jaw_l)
                set_jaw(jaw_r, [0.45, -y_close, z], [0.0, 0.0, vz], q_jaw_r)
                if jaw_shelf is not None:
                    shelf_z = pseudo_shelf_center_z + args.pseudo_jaw_lift_height * a
                    shelf_vz = (shelf_z - prev_shelf_z) / dt
                    set_jaw(jaw_shelf, [0.45, 0.0, shelf_z], [0.0, 0.0, shelf_vz])
                    prev_shelf_z = shelf_z
                sim.step()
                obj.update(1.0 / 120.0)
                jaw_l.update(1.0 / 120.0)
                jaw_r.update(1.0 / 120.0)
                if jaw_shelf is not None:
                    jaw_shelf.update(1.0 / 120.0)
                prev_z = z
            after = obj.data.root_pos_w[0].detach().cpu().tolist()
            pseudo_probe["object_pos_after_lift"] = after
            pseudo_probe["object_lift_m"] = float(after[2] - before[2])
            pseudo_probe["jaw_center_z_final"] = float(pseudo_jaw_center_z + args.pseudo_jaw_lift_height)
            obj_pos = np.asarray(after, dtype=np.float64)
            T_w_obj = pose_to_T(after, obj.data.root_quat_w[0].detach().cpu().tolist())
            # Gel frames on the inner jaw faces, +z pointing into the object.
            p_l = np.array([0.45, y_close, pseudo_jaw_center_z + args.pseudo_jaw_lift_height], dtype=np.float64)
            p_r = np.array([0.45, -y_close, pseudo_jaw_center_z + args.pseudo_jaw_lift_height], dtype=np.float64)
            n_l = R_jaw_l @ np.array([0.0, -1.0, 0.0])
            n_r = R_jaw_r @ np.array([0.0, 1.0, 0.0])
            T_w_gel_l = np.eye(4); T_w_gel_l[:3, :3] = _gel_R_from_normal(n_l); T_w_gel_l[:3, 3] = p_l + R_jaw_l @ np.array([0.0, -pseudo_jaw_thickness / 2.0, 0.0])
            T_w_gel_r = np.eye(4); T_w_gel_r[:3, :3] = _gel_R_from_normal(n_r); T_w_gel_r[:3, 3] = p_r + R_jaw_r @ np.array([0.0, pseudo_jaw_thickness / 2.0, 0.0])
            pseudo_T_gel_obj = {
                "left": (T_inv(T_w_gel_l) @ T_w_obj).tolist(),
                "right": (T_inv(T_w_gel_r) @ T_w_obj).tolist(),
            }
            for side, T in pseudo_T_gel_obj.items():
                pseudo_pen[side] = float(_pen_for_object(args.object, args.object_size, np.asarray(T), gel_grid(32, 0.009)).max())
        physics_contact_center_np = None
        if args.physics_lift_probe and args.object_at_pads and gripper_ids:
            physics_probe = {"enabled": True}
            if args.physics_place_closed_probe:
                set_gripper(args.gripper_target, 30)
                physics_contact_center_np = object_center_from_live_pads()
            else:
                set_gripper(args.gripper_open_target, 30)
                if args.physics_calibrate_close:
                    place_object_world([2.0, 2.0, 2.0])
                    physics_probe["close_calibration"] = calibrate_close_ramp(
                        args.gripper_open_target, args.gripper_target, args.physics_close_steps
                    )
                    calib = physics_probe["close_calibration"]
                    place_row = calib.get("place_row")
                    preclose_row = None
                    if place_row is None:
                        raise RuntimeError(
                            calib.get("placement_failure")
                            or "calibrated close placement did not find a contact-safe row"
                        )
                    else:
                        preclose_row = calib.get("preclose_row") or place_row
                        close_start_target = float(preclose_row["target"])
                    set_gripper(args.gripper_open_target, args.physics_preclose_steps)
                    replay_steps = int(round(float((preclose_row or {}).get("step") or args.physics_preclose_steps)))
                    replay_gripper_ramp(args.gripper_open_target, close_start_target, replay_steps)
                    set_gripper(close_start_target, 1)
                    live_place_depth = float(calib.get("placement_target_pen_m") or args.live_contact_depth)
                    physics_contact_center_np = object_center_from_live_pads(contact_depth=live_place_depth)
                    live_state = gel_gap_state()
                    live_pred = predicted_contact_for_gel_frames(physics_contact_center_np, live_state["frames"])
                    live_vals = [float(live_pred[side]) for side in ("left", "right") if side in live_pred]
                    place_row["place_center"] = physics_contact_center_np.tolist()
                    place_row["predicted_pen_m"] = live_pred
                    place_row["predicted_min_pen_m"] = float(min(live_vals)) if live_vals else 0.0
                    place_row["predicted_max_pen_m"] = float(max(live_vals)) if live_vals else 0.0
                    place_row["live_gap_m"] = live_state["gap_m"]
                    place_row["live_midpoint"] = live_state["midpoint"]
                    place_row["live_frames"] = live_state["frames"]
                    calib["place_center"] = physics_contact_center_np.tolist()
                    calib["live_place_depth_m"] = live_place_depth
                else:
                    physics_contact_center_np = object_center_from_live_pads(contact_depth=-args.physics_open_clearance)
                    close_start_target = args.gripper_open_target
            physics_probe["planned_contact_center"] = physics_contact_center_np.tolist()
            physics_probe["place_closed_probe"] = bool(args.physics_place_closed_probe)
            physics_probe["pedestal_probe"] = bool(args.physics_pedestal_probe)
            physics_probe["open_clearance_m"] = 0.0 if args.physics_place_closed_probe else float(args.physics_open_clearance)
            physics_probe["pedestal"] = place_pedestal_under(physics_contact_center_np)
            place_object_at_live_pads(physics_contact_center_np)
            placement_snapshot = current_live_contact_snapshot()
            physics_probe["after_place_precheck"] = placement_snapshot
            if args.physics_calibrate_close:
                calib = physics_probe.get("close_calibration", {})
                place_row = calib.get("place_row") or {}
                preclose_row = calib.get("preclose_row") or {}
                flog(
                    "calibrated_place "
                    f"center={physics_contact_center_np.tolist()} "
                    f"place_step={place_row.get('step')} place_target={place_row.get('target')} "
                    f"preclose_step={preclose_row.get('step')} preclose_target={preclose_row.get('target')} "
                    f"pred_place_pen={place_row.get('predicted_pen_m')} "
                    f"actual_place_pen={placement_snapshot.get('pen_m')}"
                )
            physics_probe["after_place"] = assert_placement_within_recess("physics_probe")
            physics_trace = []
            if args.physics_place_closed_probe:
                squeeze_target = squeezed_target_from(args.gripper_target, args.gripper_target)
                run_gripper_target_steps(squeeze_target, args.physics_hold_steps, "HOLD", physics_trace)
                physics_probe["close_ramp"] = {"enabled": False, "reason": "closed_place_probe"}
            else:
                physics_probe["close_ramp"] = ramp_gripper_until_contact(
                    close_start_target, args.gripper_target, args.physics_close_steps
                )
                close_final_target = float(physics_probe["close_ramp"].get("final_target", args.gripper_target))
                squeeze_target = squeezed_target_from(close_final_target, args.gripper_target)
                physics_probe["squeeze"] = {
                    "margin_rad": float(args.physics_squeeze_margin),
                    "close_final_target": close_final_target,
                    "squeeze_target": float(squeeze_target),
                }
                squeeze_rows = run_gripper_target_steps(
                    squeeze_target,
                    args.physics_squeeze_settle_steps,
                    "SQUEEZE",
                    physics_trace,
                )
                squeeze_gate = contact_gate_pass(current_contact_forces())
                physics_probe["squeeze"]["rows"] = squeeze_rows
                physics_probe["squeeze"]["force_gate_pass"] = squeeze_gate
                if squeeze_gate or not args.physics_require_force_gate:
                    hold_rows = run_gripper_target_steps(
                        squeeze_target,
                        args.physics_hold_steps,
                        "HOLD",
                        physics_trace,
                    )
                else:
                    hold_rows = []
                    physics_probe["hold_skipped_reason"] = (
                        f"force gate below {args.physics_force_gate_N:.3g} N after squeeze"
                    )
                hold_force_L = [float(r["force_N"].get("left", 0.0)) for r in hold_rows]
                hold_force_R = [float(r["force_N"].get("right", 0.0)) for r in hold_rows]
                physics_probe["hold"] = {
                    "steps_requested": int(args.physics_hold_steps),
                    "steps_run": len(hold_rows),
                    "force_gate_N": float(args.physics_force_gate_N),
                    "min_force_L_N": float(min(hold_force_L)) if hold_force_L else 0.0,
                    "min_force_R_N": float(min(hold_force_R)) if hold_force_R else 0.0,
                    "force_gate_pass": bool(
                        len(hold_rows) >= 100
                        and hold_force_L
                        and hold_force_R
                        and min(hold_force_L) >= args.physics_force_gate_N
                        and min(hold_force_R) >= args.physics_force_gate_N
                    ),
                }
            before = obj.data.root_pos_w[0].detach().cpu().tolist()
            physics_probe["object_pos_before_lift"] = before
            physics_probe["before_lift"] = current_live_contact_snapshot()
            physics_probe["robot_root_lift_method"] = hasattr(robot, "write_root_pose_to_sim")
            can_lift = bool(
                hasattr(robot, "write_root_pose_to_sim")
                and (
                    not args.physics_require_force_gate
                    or physics_probe.get("hold", {}).get("force_gate_pass", False)
                    or args.physics_place_closed_probe
                )
            )
            if can_lift:
                import torch

                root_pose0 = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=1).clone()
                lift_vz = args.physics_lift_height / max(args.physics_lift_steps, 1) / (1.0 / 120.0)
                def lift_root_pose(k):
                    frac = (k + 1) / max(args.physics_lift_steps, 1)
                    root_pose = root_pose0.clone()
                    root_pose[:, 2] += args.physics_lift_height * frac
                    root_vel = torch.zeros((1, 6), device=robot.device)
                    root_vel[:, 2] = float(lift_vz)
                    return root_pose, root_vel
                lift_rows = run_gripper_target_steps(
                    squeeze_target,
                    args.physics_lift_steps,
                    "LIFT",
                    physics_trace,
                    root_pose_fn=lift_root_pose,
                )
            else:
                lift_rows = []
                physics_probe["lift_skipped_reason"] = (
                    "force gate failed before lift" if args.physics_require_force_gate
                    else "robot root pose API unavailable"
                )
            after = obj.data.root_pos_w[0].detach().cpu().tolist()
            physics_probe["object_pos_after_lift"] = after
            physics_probe["object_lift_m"] = float(after[2] - before[2])
            lift_force_L = [float(r["force_N"].get("left", 0.0)) for r in lift_rows]
            lift_force_R = [float(r["force_N"].get("right", 0.0)) for r in lift_rows]
            lift_pen_L = [float(r["pen_m"].get("left", 0.0)) for r in lift_rows]
            lift_pen_R = [float(r["pen_m"].get("right", 0.0)) for r in lift_rows]
            physics_probe["lift"] = {
                "steps_requested": int(args.physics_lift_steps),
                "steps_run": len(lift_rows),
                "height_command_m": float(args.physics_lift_height),
                "per_step_root_delta_m": float(args.physics_lift_height / max(args.physics_lift_steps, 1)),
                "min_force_L_N": float(min(lift_force_L)) if lift_force_L else 0.0,
                "min_force_R_N": float(min(lift_force_R)) if lift_force_R else 0.0,
                "min_pen_L_m": float(min(lift_pen_L)) if lift_pen_L else 0.0,
                "min_pen_R_m": float(min(lift_pen_R)) if lift_pen_R else 0.0,
            }
            physics_probe["contact_force"] = contact_force_info
            physics_probe["passive_joint_names"] = passive_log_names
            physics_probe["trace"] = physics_trace
            physics_probe["round3_gate_pass"] = bool(
                physics_probe.get("hold", {}).get("force_gate_pass", False)
                and len(lift_rows) >= int(args.physics_lift_steps)
                and physics_probe["lift"]["min_force_L_N"] > 0.0
                and physics_probe["lift"]["min_force_R_N"] > 0.0
                and physics_probe["object_lift_m"] >= 0.08
            )
            physics_probe["p4b_gate_pass"] = bool(
                physics_probe["round3_gate_pass"]
                and physics_probe["lift"]["min_pen_L_m"] > 0.0002
                and physics_probe["lift"]["min_pen_R_m"] > 0.0002
            )
        elif args.object_at_pads and pad_indices["left"] is not None and pad_indices["right"] is not None:
            place_object_at_live_pads()
            assert_placement_within_recess("object_at_pads_initial")
            for _ in range(10):
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.write_data_to_sim()
                write_robot_root_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
            place_object_at_live_pads()
            assert_placement_within_recess("object_at_pads_final")
        else:
            for _ in range(10):
                if args.physics_lift_probe and arm_ids and arm_hold_target is not None:
                    robot.set_joint_position_target(arm_hold_target.unsqueeze(0), joint_ids=arm_ids)
                robot.write_data_to_sim()
                write_robot_root_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
        pad_pose_w = current_pad_pose_w()
        object_pose_w = {
            "pos": obj.data.root_pos_w[0].detach().cpu().tolist(),
            "quat_wxyz": obj.data.root_quat_w[0].detach().cpu().tolist(),
        }
        live_pen = {}
        live_T_gel_obj = {}
        for pad in manifest.get("pads", []):
            side = pad["name"]
            if side not in pad_pose_w:
                continue
            T_w_link = pose_to_T(pad_pose_w[side]["pos"], pad_pose_w[side]["quat_wxyz"])
            T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
            T_w_obj = pose_to_T(object_pose_w["pos"], object_pose_w["quat_wxyz"])
            T_gel_obj = T_inv(T_w_link @ T_link_gel) @ T_w_obj
            live_T_gel_obj[side] = T_gel_obj.tolist()
            live_pen[side] = float(_pen_for_object(args.object, args.object_size, T_gel_obj, gel_grid(32, 0.009)).max())
        pad_distance_m = None
        if "left" in pad_pose_w and "right" in pad_pose_w:
            pad_distance_m = float(np.linalg.norm(
                np.asarray(pad_pose_w["left"]["pos"]) - np.asarray(pad_pose_w["right"]["pos"])
            ))
        flog(f"ISAAC_SCENE_OK spawned robot/object joints={joint_names}")
        return {
            "isaac_scene": True,
            "runtime_joint_names": joint_names,
            "runtime_body_names": body_names,
            "gripper_joint_names": [joint_names[i] for i in gripper_ids],
            "gripper_control_mode": args.gripper_control_mode,
            "gripper_target": args.gripper_target,
            "runtime_joint_pos": robot.data.joint_pos[0].detach().cpu().tolist(),
            "gripper_control": gripper_control,
            "runtime_pad_pose_w": pad_pose_w,
            "runtime_pad_distance_m": pad_distance_m,
            "runtime_body_name_duplicates": duplicate_body_names,
            "runtime_pad_body_name_matches": pad_body_matches,
            "runtime_pad_indices": pad_indices,
            "gap_audit": gap_audit,
            "runtime_object_pose_w": object_pose_w,
            "runtime_T_gel_obj": pseudo_T_gel_obj or live_T_gel_obj,
            "runtime_live_pen_m": pseudo_pen or live_pen,
            "physics_probe": physics_probe,
            "pseudo_jaw_probe": pseudo_probe,
            "gripper_scan_probe": gripper_scan,
        }
    except Exception as exc:
        flog(f"ISAAC_SCENE_FAIL {type(exc).__name__}: {exc}")
        return {
            "isaac_scene": False,
            "isaac_error": f"{type(exc).__name__}: {exc}",
            "physics_probe": physics_probe,
        }


def run_demo(args) -> dict:
    Path(PROGRESS).write_text("")
    if args.object_at_pads and args.live_contact_depth > args.max_live_contact_depth:
        raise SystemExit(
            f"--live-contact-depth={args.live_contact_depth:.6g} exceeds "
            f"--max-live-contact-depth={args.max_live_contact_depth:.6g}; "
            "keep placement below the recessed gel collision surface and close the gripper dynamically."
        )
    tag = args.tag or f"{args.mode}_{int(time.time())}"
    out_dir = Path("/work/runs/sim_grasp") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    physics_pick_probe_requested = bool(args.mode == "pick" and args.physics_lift_probe)

    def abort_demo(reason: str, isaac_info: dict = None) -> None:
        isaac_info = isaac_info or {}
        summary = {
            "tag": tag,
            "mode": args.mode,
            "out_dir": str(out_dir),
            "marker": "GRASP_DEMO_FAIL",
            "failure_reason": reason,
            "physics_probe_requested": physics_pick_probe_requested,
            **isaac_info,
            "manifest_pads": manifest.get("pads", []),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        flog(f"GRASP_DEMO_FAIL {reason}")
        print("GRASP_DEMO_FAIL", out_dir / "summary.json", flush=True)
        os._exit(1)

    isaac_info = _try_start_isaac(str(manifest_path), args) if args.start_isaac else {"isaac_scene": False}
    if args.mode == "pick" and not isaac_info.get("isaac_scene"):
        abort_demo(
            f"required Isaac physics probe failed: {isaac_info.get('isaac_error', 'Isaac was not started')}",
            isaac_info,
        )
    if args.mode == "pick" and not isaac_info.get("physics_probe", {}).get("enabled", False):
        abort_demo(
            "physics probe not enabled — pass --physics-lift-probe --object-at-pads",
            isaac_info,
        )
    if physics_pick_probe_requested and not isaac_info.get("isaac_scene"):
        abort_demo(
            f"required Isaac physics probe failed: {isaac_info.get('isaac_error', 'Isaac was not started')}",
            isaac_info,
        )
    if args.object_at_pads and args.start_isaac and not isaac_info.get("isaac_scene"):
        abort_demo(
            f"Isaac object-at-pads placement failed: {isaac_info.get('isaac_error', 'unknown error')}",
            isaac_info,
        )

    sensor_l = VBTSSensor(args.ckpt, device="cuda" if args.cuda else None)
    sensor_r = VBTSSensor(args.ckpt, device="cuda" if args.cuda else None)
    grid = gel_grid(sensor_l.side, 0.009)
    tr_l = ShearTracker(mu=args.mu)
    tr_r = ShearTracker(mu=args.mu)

    phases, pens_l, pens_r, shear_l, shear_r, timings, lift_z = [], [], [], [], [], [], []
    imgs_l, imgs_r = [], []
    n_steps = args.steps
    live_T = None
    if args.live_pose_tactile and isaac_info.get("runtime_T_gel_obj"):
        live_T = {
            side: np.asarray(T, dtype=np.float64)
            for side, T in isaac_info["runtime_T_gel_obj"].items()
        }
    for step in range(n_steps):
        f = step / max(n_steps - 1, 1)
        if f < 0.25:
            phase = "HOME"
            depth = 0.0
            slide = 0.0
            lift = 0.0
        elif f < 0.55:
            phase = "CLOSE"
            depth = (f - 0.25) / 0.30 * 0.00062
            slide = 0.0
            lift = 0.0
        elif f < 0.70:
            phase = "HOLD"
            depth = 0.00062
            slide = (f - 0.55) / 0.15 * 0.00025
            lift = 0.0
        else:
            phase = "LIFT" if args.mode == "pick" else "PRESS_HOLD"
            depth = 0.00055 if args.mode == "pick" else 0.00062
            slide = 0.00025 + (f - 0.70) / 0.30 * 0.00035
            lift = (f - 0.70) / 0.30 * 0.10 if args.mode == "pick" else 0.0

        if live_T is not None and "left" in live_T and "right" in live_T:
            T_l = live_T["left"].copy()
            T_r = live_T["right"].copy()
            final_z_l = float(live_T["left"][2, 3])
            final_z_r = float(live_T["right"][2, 3])
            no_contact_z = args.object_size / 2.0 + 0.001
            if phase in ("HOME", "PRE_GRASP"):
                alpha = 0.0
            elif phase == "CLOSE":
                alpha = min(1.0, max(0.0, depth / max(args.live_contact_depth, 1e-9)))
            else:
                alpha = 1.0
            T_l[2, 3] = (1.0 - alpha) * no_contact_z + alpha * final_z_l
            T_r[2, 3] = (1.0 - alpha) * no_contact_z + alpha * final_z_r
            T_l[0, 3] += slide
            T_r[0, 3] -= slide
        else:
            # Two opposing gel frames see the object pressed into their +z visual
            # plane. The sign flip makes dot motion mirror between fingers.
            T_l = pose_to_T((slide, 0.0, args.object_size / 2.0 - depth), (1.0, 0.0, 0.0, 0.0))
            T_r = pose_to_T((-slide, 0.0, args.object_size / 2.0 - depth), (1.0, 0.0, 0.0, 0.0))
        pen_l = _pen_for_object(args.object, args.object_size, T_l, grid)
        pen_r = _pen_for_object(args.object, args.object_size, T_r, grid)
        sx_l, sy_l = tr_l.update(T_l, pen_l)
        sx_r, sy_r = tr_r.update(T_r, pen_r)

        if step % args.tactile_every == 0:
            t0 = time.perf_counter()
            out_l = sensor_l.render(pen_l, float(sx_l), float(sy_l), mu=args.mu, E=args.E, noise=not args.no_noise)
            out_r = sensor_r.render(pen_r, float(sx_r), float(sy_r), mu=args.mu, E=args.E, noise=not args.no_noise)
            timings.append(time.perf_counter() - t0)
            img_l = out_l["img"][0, 0].detach().cpu().numpy()
            img_r = out_r["img"][0, 0].detach().cpu().numpy()
            idx = len(imgs_l)
            _save_png(img_l, out_dir / f"tactile_L_{idx:04d}.png")
            _save_png(img_r, out_dir / f"tactile_R_{idx:04d}.png")
            imgs_l.append(img_l)
            imgs_r.append(img_r)
        phases.append(phase)
        pens_l.append(float(pen_l.max()))
        pens_r.append(float(pen_r.max()))
        shear_l.append([float(sx_l), float(sy_l)])
        shear_r.append([float(sx_r), float(sy_r)])
        lift_z.append(float(lift))
        if step % 20 == 0:
            flog(f"step={step} phase={phase} penL={pens_l[-1]:.6f} penR={pens_r[-1]:.6f} lift={lift:.3f}")

    npz_path = out_dir / "sequence.npz"
    physics_probe_npz = isaac_info.get("physics_probe", {}) if isinstance(isaac_info, dict) else {}
    physics_trace = physics_probe_npz.get("trace", []) if isinstance(physics_probe_npz, dict) else []
    physics_phase = np.asarray([r.get("phase", "") for r in physics_trace])
    physics_force_L = np.asarray([
        float(r.get("force_N", {}).get("left", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_force_R = np.asarray([
        float(r.get("force_N", {}).get("right", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_pen_L = np.asarray([
        float(r.get("pen_m", {}).get("left", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_pen_R = np.asarray([
        float(r.get("pen_m", {}).get("right", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_object_pos = np.asarray([
        r.get("object_pos_w", [np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_root_z = np.asarray([
        float(r.get("robot_root_z_m", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_passive_joint_pos = np.asarray([
        r.get("passive_joint_pos", [np.nan, np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 4), dtype=np.float32)
    np.savez_compressed(
        npz_path,
        pen_L=np.asarray(pens_l, dtype=np.float32),
        pen_R=np.asarray(pens_r, dtype=np.float32),
        shear_L=np.asarray(shear_l, dtype=np.float32),
        shear_R=np.asarray(shear_r, dtype=np.float32),
        lift_z=np.asarray(lift_z, dtype=np.float32),
        phase=np.asarray(phases),
        physics_phase=physics_phase,
        contact_force_N_L=physics_force_L,
        contact_force_N_R=physics_force_R,
        physics_pen_L=physics_pen_L,
        physics_pen_R=physics_pen_R,
        physics_object_pos_w=physics_object_pos,
        physics_robot_root_z_m=physics_root_z,
        passive_joint_names=np.asarray(physics_probe_npz.get("passive_joint_names", [])),
        passive_joint_pos=physics_passive_joint_pos,
        contact_force_source=np.asarray([
            physics_probe_npz.get("contact_force", {}).get("source", "none")
        ]),
        images_L=np.stack(imgs_l).astype(np.float32) if imgs_l else np.zeros((0, 160, 160), np.float32),
        images_R=np.stack(imgs_r).astype(np.float32) if imgs_r else np.zeros((0, 160, 160), np.float32),
        timings_s=np.asarray(timings, dtype=np.float32),
    )
    scan_ok = bool(
        args.gripper_scan_probe
        and isaac_info.get("isaac_scene")
        and isaac_info.get("gripper_scan_probe", {}).get("rows")
    )
    press_pen_gate_m = 0.0005
    press_pen_gate_tol_m = 1.0e-5
    press_ok = scan_ok if args.gripper_scan_probe else (
        max(pens_l) + press_pen_gate_tol_m >= press_pen_gate_m
        and max(pens_r) + press_pen_gate_tol_m >= press_pen_gate_m
    )
    physical_lift_ok = bool(
        args.mode == "pick"
        and isaac_info.get("isaac_scene")
        and physics_probe_npz.get("enabled", False)
        and physics_force_L.size > 0
        and physics_force_R.size > 0
        and np.isfinite(physics_force_L).any()
        and np.isfinite(physics_force_R).any()
        and (
            not args.physics_require_force_gate
            or bool(physics_probe_npz.get("hold", {}).get("force_gate_pass", False))
        )
        and (
            physics_probe_npz.get("p4b_gate_pass", False)
        )
    )
    pick_failure_reason = None
    if args.mode == "pick":
        if not physics_probe_npz.get("enabled", False):
            pick_failure_reason = "physics probe not enabled — pass --physics-lift-probe --object-at-pads"
        elif physics_force_L.size == 0 or physics_force_R.size == 0:
            pick_failure_reason = "physics probe produced empty contact force arrays"
        elif not np.isfinite(physics_force_L).any() or not np.isfinite(physics_force_R).any():
            pick_failure_reason = "physics probe contact force arrays contain no finite values"
        elif args.physics_require_force_gate and not physics_probe_npz.get("hold", {}).get("force_gate_pass", False):
            pick_failure_reason = "physics probe force gate failed"
        elif not physics_probe_npz.get("p4b_gate_pass", False):
            pick_failure_reason = "physics probe P4b lift gate failed"
    pick_ok = physical_lift_ok
    if args.mode == "pick":
        marker = "GRASP_DEMO_OK" if pick_ok else "GRASP_DEMO_FAIL"
    else:
        marker = "PRESS_OK" if press_ok else "GRASP_DEMO_FAIL"
    representative_idx = max(0, len(imgs_l) - 1)
    summary = {
        "tag": tag,
        "mode": args.mode,
        "out_dir": str(out_dir),
        "sequence": str(npz_path),
        "representative_tactile_L": str(out_dir / f"tactile_L_{representative_idx:04d}.png"),
        "representative_tactile_R": str(out_dir / f"tactile_R_{representative_idx:04d}.png"),
        "max_pen_L_m": float(max(pens_l)),
        "max_pen_R_m": float(max(pens_r)),
        "press_pen_gate_m": press_pen_gate_m,
        "press_pen_gate_tol_m": press_pen_gate_tol_m,
        "final_lift_m": float(lift_z[-1]),
        "mean_tactile_ms": float(np.mean(timings) * 1e3) if timings else None,
        "live_pose_tactile": live_T is not None,
        "round3_gate_pass": bool(physics_probe_npz.get("round3_gate_pass", False)),
        "p4b_gate_pass": bool(physics_probe_npz.get("p4b_gate_pass", False)),
        "contact_force_source": physics_probe_npz.get("contact_force", {}).get("source", "none"),
        "marker": marker,
        **isaac_info,
        "manifest_pads": manifest.get("pads", []),
    }
    if pick_failure_reason is not None and marker != "GRASP_DEMO_OK":
        summary["failure_reason"] = pick_failure_reason
    def finite_json(x):
        if isinstance(x, dict):
            return {k: finite_json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [finite_json(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x

    (out_dir / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    flog(marker)
    print(marker, npz_path, flush=True)
    os._exit(0 if marker in ("GRASP_DEMO_OK", "PRESS_OK") else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["press", "pick"], default="press")
    ap.add_argument("--object", choices=["sphere", "cube", "cylinder"], default="sphere")
    ap.add_argument("--object-size", type=float, default=0.015)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--tactile-every", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--E", type=float, default=1.0e5)
    ap.add_argument("--ckpt", default="/work/data/assets/vbts_fno/lr_fno_realistic_bc.pt")
    ap.add_argument("--manifest", default="/work/data/assets/ur3e_robotiq_vbts.json")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--start-isaac", action="store_true", default=True)
    ap.add_argument("--robot-z", type=float, default=0.0)
    ap.add_argument("--robot-root-rpy-deg", default="0,0,0")
    ap.add_argument("--gripper-target", type=float, default=0.35)
    ap.add_argument("--gripper-open-target", type=float, default=0.0)
    ap.add_argument("--gripper-control-mode", choices=["finger", "all"], default="finger")
    ap.add_argument("--gripper-stiffness", type=float, default=60.0)
    ap.add_argument("--gripper-damping", type=float, default=8.0)
    ap.add_argument("--gripper-effort-limit", type=float, default=5.0)
    ap.add_argument("--passive-gripper-stiffness", type=float, default=0.0)
    ap.add_argument("--passive-gripper-damping", type=float, default=0.0)
    ap.add_argument("--passive-gripper-effort-limit", type=float, default=0.0)
    ap.add_argument("--robot-solver-position-iters", type=int, default=32)
    ap.add_argument("--robot-solver-velocity-iters", type=int, default=8)
    ap.add_argument("--robot-max-depenetration-velocity", type=float, default=0.2)
    ap.add_argument("--gripper-scan-probe", action="store_true")
    ap.add_argument("--gripper-scan-targets", default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35")
    ap.add_argument("--gripper-scan-steps", type=int, default=30)
    ap.add_argument("--gripper-scan-stable-joint-abs", type=float, default=10.0)
    ap.add_argument("--gap-audit", action="store_true")
    ap.add_argument("--gap-audit-targets", default="0.0,0.06,0.12,0.18,0.24")
    ap.add_argument("--gap-audit-settle-steps", type=int, default=30)
    ap.add_argument("--object-at-pads", action="store_true")
    ap.add_argument("--object-pad-z-offset", type=float, default=0.0)
    ap.add_argument("--object-center-solver", choices=["average", "lsq"], default="lsq")
    ap.add_argument("--object-center-tangent-weight", type=float, default=0.0)
    ap.add_argument("--object-center-tangent-x-weight", type=float, default=-1.0)
    ap.add_argument("--object-center-tangent-y-weight", type=float, default=-1.0)
    ap.add_argument("--object-center-offset-world", default="0,0,0")
    ap.add_argument("--live-contact-depth", type=float, default=0.0004)
    ap.add_argument("--place-target-pen", type=float, default=0.00015)
    ap.add_argument("--max-live-contact-depth", type=float, default=0.0005)
    ap.add_argument("--max-live-placement-pen", type=float, default=0.0006)
    ap.add_argument("--live-pose-tactile", action="store_true")
    ap.add_argument("--object-mu", type=float, default=0.9)
    ap.add_argument("--object-mass", type=float, default=0.02)
    ap.add_argument("--friction-combine-mode", choices=["average", "min", "multiply", "max"], default="average")
    ap.add_argument("--object-solver-position-iters", type=int, default=-1)
    ap.add_argument("--object-solver-velocity-iters", type=int, default=-1)
    ap.add_argument("--object-max-depenetration-velocity", type=float, default=-1.0)
    ap.add_argument("--object-contact-offset", type=float, default=2.0e-4)
    ap.add_argument("--object-rest-offset", type=float, default=0.0)
    ap.add_argument("--physics-lift-probe", action="store_true")
    ap.add_argument("--physics-place-closed-probe", action="store_true")
    ap.add_argument("--physics-pedestal-probe", action="store_true")
    ap.add_argument("--physics-pedestal-size", type=float, default=0.04)
    ap.add_argument("--physics-pedestal-height", type=float, default=0.30)
    ap.add_argument("--physics-hold-steps", type=int, default=100)
    ap.add_argument("--physics-close-steps", type=int, default=180)
    ap.add_argument("--physics-close-stop-pen", type=float, default=0.00055)
    ap.add_argument("--physics-squeeze-margin", type=float, default=0.02)
    ap.add_argument("--physics-squeeze-settle-steps", type=int, default=20)
    ap.add_argument("--physics-force-gate-N", type=float, default=1.0)
    ap.add_argument("--physics-require-force-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--physics-open-clearance", type=float, default=0.002)
    ap.add_argument("--physics-calibrate-close", action="store_true")
    ap.add_argument("--physics-preclose-clearance", type=float, default=0.010)
    ap.add_argument("--physics-preclose-steps", type=int, default=30)
    ap.add_argument("--physics-close-stall-eps", type=float, default=1.0e-5)
    ap.add_argument("--physics-close-stall-steps", type=int, default=20)
    ap.add_argument("--physics-close-log-every", type=int, default=20)
    ap.add_argument("--physics-lift-steps", type=int, default=120)
    ap.add_argument("--physics-lift-height", type=float, default=0.08)
    ap.add_argument("--pseudo-jaw-probe", action="store_true")
    ap.add_argument("--pseudo-jaw-z", type=float, default=0.0)
    ap.add_argument("--pseudo-jaw-open-half-gap", type=float, default=0.0)
    ap.add_argument("--pseudo-jaw-close-half-gap", type=float, default=0.0)
    ap.add_argument("--pseudo-jaw-indent", type=float, default=0.0015)
    ap.add_argument("--pseudo-jaw-pitch-deg", type=float, default=0.0)
    ap.add_argument("--pseudo-jaw-support-probe", action="store_true")
    ap.add_argument("--pseudo-jaw-preclose-settle-steps", type=int, default=-1)
    ap.add_argument("--pseudo-jaw-close-steps", type=int, default=80)
    ap.add_argument("--pseudo-jaw-lift-steps", type=int, default=120)
    ap.add_argument("--pseudo-jaw-lift-height", type=float, default=0.09)
    args = ap.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
