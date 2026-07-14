#!/usr/bin/env python3
"""Container-only grasp/press demo driver.

This driver keeps Isaac startup and asset spawning in the loop, while the tactile
rendering path is the same host-testable VBTSSensor used by unit smoke tests.
The robot control is intentionally manifest-driven and open-loop so P0/P3 can
pin real joint names before later IK tuning.
"""
from __future__ import annotations

import glob
import json
import math
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

from novbts.simulation.runtime.gel_contact import (
    ShearTracker,
    T_inv,
    gel_grid,
    pen_cuboid,
    pen_cylinder,
    pen_sphere,
    pose_to_T,
)
from novbts.simulation.runtime.vbts_sensor import VBTSSensor
from novbts.simulation.runtime.grasp_output import (
    PROGRESS,
    flog,
    save_png as _save_png,
    save_rgb_png as _save_rgb_png,
)
from novbts.simulation.runtime.grasp_math import (
    Rx as _Rx,
    Ry as _Ry,
    Rz as _Rz,
    float_list_arg as _float_list_arg,
    gel_R_from_normal as _gel_R_from_normal,
    quat_angle_error_wxyz as _quat_angle_error_wxyz,
    quat_between_vectors_wxyz as _quat_between_vectors_wxyz,
    quat_conj_wxyz as _quat_conj_wxyz,
    quat_from_rpy_deg as _quat_from_rpy_deg,
    quat_mul_wxyz as _quat_mul_wxyz,
    quat_wxyz_from_R as _quat_wxyz_from_R,
    vec3_arg as _vec3_arg,
)
from novbts.simulation.runtime.grasp_tactile_util import (
    pen_for_object as _pen_for_object,
    pen_grid_stats as _pen_grid_stats,
    support_extent_for_object as _support_extent_for_object,
)
from novbts.simulation.runtime.grasp_geometry import (
    T_from_gel_frame as _T_from_gel_frame,
    gel_gap_from_frames,
    grip_axis_for_gel_frames,
    object_center_for_opposing_normals,
    pen_max_values,
    pen_stat_values,
    placement_balance_tolerance,
    placement_pen_summary,
)
from novbts.simulation.runtime.grasp_livestream import frame_viewport_on_robot, hold_livestream
from novbts.simulation.runtime.grasp_cli import build_arg_parser
from novbts.simulation.runtime.grasp_tactile_overlay import TactileViewportOverlay


_LIVE_SIMULATION_APP = None


def _try_start_isaac(manifest_path: str, args) -> dict:
    """Spawn a minimal scene. Failures are logged but do not hide tactile output."""
    global _LIVE_SIMULATION_APP
    physics_probe = {"enabled": False}
    livestream_value = int(getattr(args, "livestream", 0))
    livestream_info = {"requested": bool(livestream_value), "value": livestream_value}
    try:
        import inspect
        from isaaclab.app import AppLauncher

        # Livestream needs the RTX-rendering viewport to be active so the streaming
        # client has frames to capture. With enable_cameras=False the app resolves to
        # the non-rendering headless experience: the control channel connects (RTP
        # ping succeeds) but the client receives no video payload -> black screen.
        # Force rendering on for any livestream mode (1=native, 2=webrtc).
        launcher_kwargs = {
            "headless": True,
            "enable_cameras": bool(args.world_camera) or livestream_value > 0,
        }
        if livestream_value:
            app_launcher_sig = inspect.signature(AppLauncher.__init__)
            livestream_info["app_launcher_signature"] = str(app_launcher_sig)
            params = app_launcher_sig.parameters
            supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if "livestream" not in params and not supports_kwargs:
                raise RuntimeError(
                    "AppLauncher.__init__ does not expose livestream support: "
                    f"{app_launcher_sig}"
                )
            launcher_kwargs["livestream"] = livestream_value
            livestream_info["api"] = "AppLauncher(headless=True, enable_cameras=..., livestream=<int>)"
            # Native streaming (mode 1) captures the *presented* viewport and encodes
            # it via NVENC. The isaaclab `rendering.kit` experience ships with
            # `exts."omni.renderer.core".present.enabled=false` (line 62) "to improve
            # performance", which keeps the present thread down and NVENC at 0% -> the
            # client connects (port 48010 established, RTP ping OK) but never receives a
            # video payload -> black screen. Re-enable the present thread at kit startup
            # via a `--/` setting override so NVENC has presented frames to encode. We
            # keep rendering.kit itself (old_streaming.kit is deprecated in this 4.5
            # image and its rendering_modes/balanced.kit is missing -> scene build
            # fails, SimulationApp never comes up).
            if livestream_value == 1:
                present_override = "--/exts/omni.renderer.core/present/enabled=true"
                if present_override not in sys.argv:
                    sys.argv.append(present_override)
                livestream_info["present_override"] = present_override
        app_launcher = AppLauncher(**launcher_kwargs)
        _simulation_app = app_launcher.app
        _LIVE_SIMULATION_APP = _simulation_app
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
        effective_gripper_stiffness = float(args.gripper_stiffness)
        effective_gripper_damping = float(args.gripper_damping)
        effective_gripper_effort_limit = float(args.gripper_effort_limit)
        if args.grasp_mode == "ik":
            if float(args.ik_close_gripper_stiffness) >= 0.0:
                effective_gripper_stiffness = float(args.ik_close_gripper_stiffness)
            if float(args.ik_close_gripper_damping) >= 0.0:
                effective_gripper_damping = float(args.ik_close_gripper_damping)
            if float(args.ik_close_gripper_effort_limit) >= 0.0:
                effective_gripper_effort_limit = float(args.ik_close_gripper_effort_limit)
        actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
                                       stiffness=400.0, damping=80.0),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=gripper_joint_expr,
                stiffness=effective_gripper_stiffness,
                damping=effective_gripper_damping,
                effort_limit=effective_gripper_effort_limit,
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
        init_joint_pos = None
        if args.grasp_mode == "ik":
            ready = _float_list_arg(args.ik_arm_ready_joints, 6)
            init_joint_pos = {
                "shoulder_pan_joint": ready[0],
                "shoulder_lift_joint": ready[1],
                "elbow_joint": ready[2],
                "wrist_1_joint": ready[3],
                "wrist_2_joint": ready[4],
                "wrist_3_joint": ready[5],
                "finger_joint": float(args.gripper_open_target),
            }
        init_state_kwargs = {
            "pos": (args.robot_x, args.robot_y, args.robot_z),
            "rot": tuple(_quat_from_rpy_deg(args.robot_root_rpy_deg)),
        }
        if init_joint_pos is not None:
            init_state_kwargs["joint_pos"] = init_joint_pos
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/work/data/assets/ur3e_robotiq_vbts.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**robot_rigid_kwargs) if robot_rigid_kwargs else None,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(**init_state_kwargs),
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
        effective_object_max_depenetration_velocity = float(args.object_max_depenetration_velocity)
        if (
            effective_object_max_depenetration_velocity < 0.0
            and args.physics_lift_probe
            and args.grasp_mode != "ik"
            and args.object_at_pads
            and args.physics_object_max_depenetration_velocity >= 0.0
        ):
            effective_object_max_depenetration_velocity = float(args.physics_object_max_depenetration_velocity)
        if args.object_solver_position_iters >= 0:
            rigid_kwargs["solver_position_iteration_count"] = args.object_solver_position_iters
        if args.object_solver_velocity_iters >= 0:
            rigid_kwargs["solver_velocity_iteration_count"] = args.object_solver_velocity_iters
        if effective_object_max_depenetration_velocity >= 0.0:
            rigid_kwargs["max_depenetration_velocity"] = effective_object_max_depenetration_velocity
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
        scene_pedestal_height = float(args.physics_pedestal_height)
        scene_pedestal_height_source = "physics_pedestal_height"
        if args.grasp_mode == "ik" and float(args.ik_pedestal_height) > 0.0:
            scene_pedestal_height = float(args.ik_pedestal_height)
            scene_pedestal_height_source = "ik_pedestal_height"
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
        if args.physics_pedestal_probe or args.grasp_mode == "ik":
            pedestal_spawn = sim_utils.CuboidCfg(
                size=(args.physics_pedestal_size, args.physics_pedestal_size, scene_pedestal_height),
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
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, scene_pedestal_height / 2.0)),
            ))
        world_camera = None
        world_camera_frames = []
        world_camera_info = {
            "enabled": bool(args.world_camera),
            "frames": world_camera_frames,
        }
        if args.world_camera:
            try:
                from isaaclab.sensors import Camera, CameraCfg

                cam_cfg = CameraCfg(
                    prim_path="/World/WorldCamera",
                    update_period=0.0,
                    height=int(args.world_camera_height),
                    width=int(args.world_camera_width),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(args.world_camera_focal_length),
                        focus_distance=float(args.world_camera_focus_distance),
                        horizontal_aperture=float(args.world_camera_horizontal_aperture),
                    ),
                )
                world_camera = Camera(cam_cfg)
                world_camera_info.update({
                    "source": "isaaclab.sensors.Camera",
                    "prim_path": cam_cfg.prim_path,
                    "width": int(args.world_camera_width),
                    "height": int(args.world_camera_height),
                    "data_types": list(cam_cfg.data_types),
                })
            except Exception as exc:
                world_camera = None
                world_camera_info.update({
                    "enabled": False,
                    "setup_error": f"{type(exc).__name__}: {exc}",
                })
                flog(f"WORLD_CAMERA_SETUP_FAIL {type(exc).__name__}: {exc}")
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

        # When livestreaming, auto-frame the streamed viewport camera on the robot
        # (the default Perspective camera looks at the origin and clips the arm).
        robot_center_w = None
        robot_diag = 0.0
        if int(getattr(args, "livestream", 0)) > 0 or bool(getattr(args, "tactile_overlay", False)):
            robot_center_w, robot_diag = frame_viewport_on_robot()
        tactile_overlay = None
        if bool(getattr(args, "tactile_overlay", False)):
            tactile_overlay = TactileViewportOverlay(
                enabled=True, scale_m=float(args.tactile_overlay_scale),
                anchor=args.tactile_overlay_anchor,
                update_every=int(args.tactile_overlay_update_every),
                output_dir=getattr(args, "_out_dir", "/work/runs/sim_grasp"),
                camera_eye=_vec3_arg(args.world_camera_eye),
                camera_target=_vec3_arg(args.world_camera_target),
            )
            tactile_overlay.setup(robot_center_w, robot_diag=robot_diag)
            frame_viewport_on_robot()

        def configure_world_camera_pose():
            if world_camera is None:
                return
            import torch

            eye = _vec3_arg(args.world_camera_eye)
            target = _vec3_arg(args.world_camera_target)
            device = getattr(world_camera, "device", robot.device)
            eyes = torch.tensor(eye, device=device, dtype=torch.float32).reshape(1, 3)
            targets = torch.tensor(target, device=device, dtype=torch.float32).reshape(1, 3)
            if hasattr(world_camera, "set_world_poses_from_view"):
                world_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
                world_camera_info["pose_api"] = "set_world_poses_from_view"
            else:
                forward = target - eye
                forward = forward / (np.linalg.norm(forward) + 1.0e-12)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                right = np.cross(forward, up)
                if np.linalg.norm(right) < 1.0e-9:
                    right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                right = right / (np.linalg.norm(right) + 1.0e-12)
                up = np.cross(right, forward)
                up = up / (np.linalg.norm(up) + 1.0e-12)
                # USD camera local axes are +X right, +Y up, -Z forward.
                R = np.column_stack([right, up, -forward])
                quat = torch.tensor(
                    _quat_wxyz_from_R(R),
                    device=device,
                    dtype=torch.float32,
                ).reshape(1, 4)
                world_camera.set_world_poses(positions=eyes, orientations=quat, convention="world")
                world_camera_info["pose_api"] = "set_world_poses"
            world_camera_info["eye_w"] = eye.tolist()
            world_camera_info["target_w"] = target.tolist()

        def world_camera_capture_label(phase: str, step_in_phase: int) -> str | None:
            if not args.world_camera:
                return None
            if phase == "PRE_GRASP" and step_in_phase >= max(1, int(args.ik_pregrasp_steps)):
                return "pre_grasp"
            if phase == "SETTLE" and step_in_phase >= min(20, max(1, int(args.physics_hold_steps))):
                return "close_clamp"
            if phase == "LIFT" and step_in_phase >= max(1, int(args.physics_lift_steps)):
                return "lift_peak"
            return None

        world_camera_captured = set()

        def capture_world_camera_frame(label: str, row: dict) -> None:
            if world_camera is None or label in world_camera_captured:
                return
            try:
                if hasattr(sim, "render"):
                    sim.render()
                world_camera.update(1.0 / 120.0)
                rgb = world_camera.data.output.get("rgb")
                if rgb is None:
                    raise RuntimeError("Camera output did not contain 'rgb'")
                if hasattr(rgb, "detach"):
                    rgb = rgb.detach().cpu().numpy()
                path = Path(getattr(args, "_out_dir", "/work/runs/sim_grasp")) / (
                    f"world_main_{label}_{int(row.get('step_in_phase', 0)):04d}.png"
                )
                _save_rgb_png(rgb, path)
                frame = {
                    "label": label,
                    "phase": row.get("phase"),
                    "step_in_phase": int(row.get("step_in_phase") or 0),
                    "path": str(path),
                    "view": "main",
                }
                world_camera_frames.append(frame)
                world_camera_captured.add(label)
                flog(f"WORLD_CAMERA_FRAME {label} {path}")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                world_camera_info.setdefault("capture_errors", []).append({
                    "label": label,
                    "phase": row.get("phase"),
                    "step_in_phase": int(row.get("step_in_phase") or 0),
                    "error": err,
                })
                world_camera_captured.add(label)
                flog(f"WORLD_CAMERA_CAPTURE_FAIL {label} {err}")

        configure_world_camera_pose()
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

        def object_center_from_gel_frames(frames, contact_depth):
            origins = []
            normals = []
            supports = []
            desired_offsets = []
            for side in ("left", "right"):
                frame = frames.get(side)
                if frame is None:
                    continue
                T_w_gel = _T_from_gel_frame(frame)
                normal = T_w_gel[:3, 2].copy()
                support = _support_extent_for_object(args.object, args.object_size, normal)
                desired = support - float(contact_depth)
                origins.append(T_w_gel[:3, 3].copy())
                normals.append(normal)
                supports.append(support)
                desired_offsets.append(desired)
            if args.object_center_solver == "lsq":
                balanced = object_center_for_opposing_normals(origins, normals, supports)
                if balanced is not None:
                    return balanced
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

        def predicted_contact_stats_for_gel_frames(center_np, frames, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
            T_w_obj = pose_to_T(np.asarray(center_np, dtype=np.float64).tolist(), quat_wxyz)
            pen_stats = {}
            for side, frame in frames.items():
                T = T_inv(_T_from_gel_frame(frame)) @ T_w_obj
                pen_grid = _pen_for_object(args.object, args.object_size, T, gel_grid(32, 0.009))
                pen_stats[side] = _pen_grid_stats(pen_grid)
            return pen_stats

        def predicted_contact_for_gel_frames(center_np, frames, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
            return {
                side: float(stats["stat_m"])
                for side, stats in predicted_contact_stats_for_gel_frames(
                    center_np, frames, quat_wxyz
                ).items()
            }

        def balance_center_for_gel_frames(center_np, frames, target_depth, max_shift_m=None):
            center = np.asarray(center_np, dtype=np.float64)
            axis = grip_axis_for_gel_frames(frames)
            if axis is None:
                pen_stats = predicted_contact_stats_for_gel_frames(center, frames)
                pen = pen_stat_values(pen_stats)
                return center, {
                    "enabled": False,
                    "reason": "missing_opposing_gel_axis",
                    "pen_m": pen,
                    "pen_stat_m": pen,
                    "pen_max_m": pen_max_values(pen_stats),
                    "pen_stats": pen_stats,
                    **placement_pen_summary(pen, target_depth),
                }
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            cap = safe_grip_pen_limit(recess)
            tol = placement_balance_tolerance(target_depth)
            max_shift = (
                2.0 * max(float(target_depth), recess, 1.0e-4)
                if max_shift_m is None else float(max_shift_m)
            )
            max_shift = float(min(max(max_shift, 2.5e-4), 1.5e-3))
            candidates = []
            for span, count in ((max_shift, 61), (max_shift / 10.0, 41), (max_shift / 100.0, 41)):
                if candidates:
                    best_shift = float(min(candidates, key=lambda item: item[0])[1])
                    shifts = np.linspace(best_shift - span, best_shift + span, count)
                else:
                    shifts = np.linspace(-span, span, count)
                for shift in shifts:
                    shift = float(min(max(shift, -max_shift), max_shift))
                    trial = center + axis * shift
                    pen_stats = predicted_contact_stats_for_gel_frames(trial, frames)
                    pen = pen_stat_values(pen_stats)
                    pen_max = pen_max_values(pen_stats)
                    stats = placement_pen_summary(pen, target_depth)
                    if stats["mean_pen_m"] is None:
                        continue
                    over_cap = max(0.0, float(stats["max_pen_m"]) - cap)
                    under_recess = max(0.0, recess - float(stats["min_pen_m"]))
                    mean_error = abs(float(stats["mean_pen_m"]) - float(target_depth))
                    imbalance = float(stats["imbalance_m"] or 0.0)
                    raw_max_pen = max(pen_max.values()) if pen_max else 0.0
                    clamp_penalty = max(0.0, float(raw_max_pen) - 7.4e-4)
                    objective = (
                        4.0 * imbalance
                        + 2.0 * mean_error
                        + 25.0 * over_cap
                        + 8.0 * under_recess
                        + 10.0 * clamp_penalty
                    )
                    candidates.append((objective, shift, trial, pen, pen_max, pen_stats, stats))
            if not candidates:
                pen_stats = predicted_contact_stats_for_gel_frames(center, frames)
                pen = pen_stat_values(pen_stats)
                return center, {
                    "enabled": False,
                    "reason": "no_finite_balance_candidates",
                    "pen_m": pen,
                    "pen_stat_m": pen,
                    "pen_max_m": pen_max_values(pen_stats),
                    "pen_stats": pen_stats,
                    **placement_pen_summary(pen, target_depth),
                }
            objective, shift, best_center, pen, pen_max, pen_stats, stats = min(candidates, key=lambda item: item[0])
            return best_center, {
                "enabled": True,
                "method": "grip_axis_penalty_search",
                "axis_w": axis.tolist(),
                "shift_m": float(shift),
                "max_shift_m": float(max_shift),
                "target_pen_m": float(target_depth),
                "recess_m": float(recess),
                "cap_m": float(cap),
                "tolerance_m": float(tol),
                "objective": float(objective),
                "pen_m": pen,
                "pen_stat_m": pen,
                "pen_max_m": pen_max,
                "pen_stats": pen_stats,
                "pass": bool(
                    stats["imbalance_m"] is not None
                    and float(stats["imbalance_m"]) <= tol
                    and float(stats["min_pen_m"]) > recess
                    and float(stats["max_pen_m"]) < cap
                ),
                **stats,
            }

        def placement_prediction_for_gel_frames(frames, contact_depth):
            if not frames or "left" not in frames or "right" not in frames:
                return None
            base_center = object_center_from_gel_frames(frames, contact_depth)
            center, balance = balance_center_for_gel_frames(base_center, frames, contact_depth)
            pred_stats = predicted_contact_stats_for_gel_frames(center, frames)
            pred = pen_stat_values(pred_stats)
            pred_max = pen_max_values(pred_stats)
            stats = placement_pen_summary(pred, contact_depth)
            if stats["mean_pen_m"] is None:
                return None
            target = float(contact_depth)
            return {
                "center": center.tolist(),
                "base_center": base_center.tolist(),
                "center_balance": balance,
                "pen_m": pred,
                "pen_stat_m": pred,
                "pen_max_m": pred_max,
                "pen_stats": pred_stats,
                "min_pen_m": stats["min_pen_m"],
                "max_pen_m": stats["max_pen_m"],
                "mean_pen_m": stats["mean_pen_m"],
                "target_pen_m": target,
                "pen_error_m": stats["pen_error_m"],
                "mean_pen_delta_m": stats["mean_pen_delta_m"],
                "imbalance_m": stats["imbalance_m"],
            }

        def nominal_gap_for_contact_depth(contact_depth):
            return float(max(float(args.object_size) - 2.0 * float(contact_depth), 0.0))

        def support_required_gap_for_gel_frames(frames, contact_depth):
            required = 0.0
            supports = {}
            if not frames:
                return None, supports
            for side in ("left", "right"):
                frame = frames.get(side)
                if frame is None:
                    return None, supports
                normal = np.asarray(frame["normal"], dtype=np.float64)
                support = _support_extent_for_object(args.object, args.object_size, normal)
                supports[side] = float(support)
                required += support - float(contact_depth)
            return float(max(required, 0.0)), supports

        def required_gap_for_gel_frames(frames, contact_depth):
            _, supports = support_required_gap_for_gel_frames(frames, contact_depth)
            return nominal_gap_for_contact_depth(contact_depth), supports

        def current_live_contact_snapshot():
            pad_pose = current_pad_pose_w()
            gel_frames = current_gel_frames_w(pad_pose)
            obj_pose = {
                "pos": obj.data.root_pos_w[0].detach().cpu().tolist(),
                "quat_wxyz": obj.data.root_quat_w[0].detach().cpu().tolist(),
            }
            T_w_obj = pose_to_T(obj_pose["pos"], obj_pose["quat_wxyz"])
            pen = {}
            pen_stat = {}
            pen_p90 = {}
            pen_mean_contact = {}
            pen_contact_fraction = {}
            T_gel_obj = {}
            for pad in manifest.get("pads", []):
                side = pad["name"]
                if side not in pad_pose:
                    continue
                T_w_link = pose_to_T(pad_pose[side]["pos"], pad_pose[side]["quat_wxyz"])
                T_link_gel = np.asarray(pad["T_link_gel"], dtype=np.float64)
                T = T_inv(T_w_link @ T_link_gel) @ T_w_obj
                T_gel_obj[side] = T.tolist()
                pen_grid = _pen_for_object(args.object, args.object_size, T, gel_grid(32, 0.009))
                stats = _pen_grid_stats(pen_grid)
                pen[side] = float(stats["max_m"])
                pen_stat[side] = float(stats["stat_m"])
                pen_p90[side] = float(stats["p90_m"])
                pen_mean_contact[side] = float(stats["mean_contact_m"])
                pen_contact_fraction[side] = float(stats["contact_fraction"])
            return {
                "object_pose_w": obj_pose,
                "pad_pose_w": pad_pose,
                "gel_origin_w": current_gel_origins_w(pad_pose),
                "gel_frame_w": gel_frames,
                "gel_gap": gel_gap_from_frames(gel_frames),
                "T_gel_obj": T_gel_obj,
                "pen_m": pen,
                "pen_stat_m": pen_stat,
                "pen_max_m": pen,
                "pen_p90_m": pen_p90,
                "pen_mean_contact_m": pen_mean_contact,
                "pen_contact_fraction": pen_contact_fraction,
            }

        def grip_symmetry_from_snapshot(snap):
            obj_pos = np.asarray(snap.get("object_pose_w", {}).get("pos", [np.nan, np.nan, np.nan]), dtype=np.float64)
            gel = snap.get("gel_origin_w", {})
            pad = snap.get("pad_pose_w", {})
            frames = snap.get("gel_frame_w", {})
            if "left" not in gel or "right" not in gel or not np.all(np.isfinite(obj_pos)):
                return {
                    "axis_w": [np.nan, np.nan, np.nan],
                    "gel_midpoint_w": [np.nan, np.nan, np.nan],
                    "object_minus_gel_midpoint_w": [np.nan, np.nan, np.nan],
                    "object_axial_offset_m": None,
                    "gel_origin_separation_m": None,
                    "left_gel_to_object_m": None,
                    "object_to_right_gel_m": None,
                    "pad_separation_m": None,
                    "object_pad_axial_offset_m": None,
                }
            left = np.asarray(gel["left"], dtype=np.float64)
            right = np.asarray(gel["right"], dtype=np.float64)
            sep = float(np.linalg.norm(right - left))
            axis = right - left
            if "left" in frames and "right" in frames:
                left_n = np.asarray(frames["left"].get("normal", [np.nan, np.nan, np.nan]), dtype=np.float64)
                right_n = np.asarray(frames["right"].get("normal", [np.nan, np.nan, np.nan]), dtype=np.float64)
                if np.all(np.isfinite(left_n)) and np.all(np.isfinite(right_n)):
                    left_n = left_n / (np.linalg.norm(left_n) + 1.0e-12)
                    right_n = right_n / (np.linalg.norm(right_n) + 1.0e-12)
                    opposing_axis = left_n - right_n
                    if np.linalg.norm(opposing_axis) > 1.0e-9:
                        axis = opposing_axis
            axis = axis / (np.linalg.norm(axis) + 1.0e-12)
            if float((right - left) @ axis) < 0.0:
                axis = -axis
            origin_mid = 0.5 * (left + right)
            face_mid_coord = 0.5 * (float(axis @ left) + float(axis @ right))
            mid = origin_mid + axis * (face_mid_coord - float(axis @ origin_mid))
            obj_delta = obj_pos - mid
            out = {
                "axis_w": axis.tolist(),
                "gel_midpoint_w": mid.tolist(),
                "object_minus_gel_midpoint_w": obj_delta.tolist(),
                "object_axial_offset_m": float(obj_delta @ axis),
                "gel_origin_separation_m": sep,
                "left_gel_to_object_m": float((obj_pos - left) @ axis),
                "object_to_right_gel_m": float((right - obj_pos) @ axis),
                "pad_separation_m": None,
                "object_pad_axial_offset_m": None,
            }
            if "left" in pad and "right" in pad:
                left_pad = np.asarray(pad["left"].get("pos", [np.nan, np.nan, np.nan]), dtype=np.float64)
                right_pad = np.asarray(pad["right"].get("pos", [np.nan, np.nan, np.nan]), dtype=np.float64)
                pad_axis = right_pad - left_pad
                pad_sep = float(np.linalg.norm(pad_axis))
                pad_axis = pad_axis / (pad_sep + 1.0e-12)
                pad_mid = 0.5 * (left_pad + right_pad)
                out["pad_separation_m"] = pad_sep
                out["object_pad_axial_offset_m"] = float((obj_pos - pad_mid) @ pad_axis)
            return out

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

        def effective_contact_mu():
            gel_mu = float(manifest.get("gel", {}).get("mu", args.object_mu))
            object_mu = float(args.object_mu)
            mode = str(args.friction_combine_mode)
            if mode == "min":
                return min(gel_mu, object_mu)
            if mode == "multiply":
                return gel_mu * object_mu
            if mode == "max":
                return max(gel_mu, object_mu)
            return 0.5 * (gel_mu + object_mu)

        def friction_margin_from_forces(force_snapshot):
            forces = force_snapshot.get("forces", {})
            normal_l = float(forces.get("left", {}).get("normal_N", 0.0))
            normal_r = float(forces.get("right", {}).get("normal_N", 0.0))
            mu_eff = effective_contact_mu()
            weight = float(args.object_mass) * 9.81
            normal_total = max(normal_l, 0.0) + max(normal_r, 0.0)
            capacity = mu_eff * normal_total
            required_total = float("inf") if mu_eff <= 0.0 else weight / mu_eff
            required_per_finger = required_total / 2.0
            return {
                "mu_eff": float(mu_eff),
                "object_mass_kg": float(args.object_mass),
                "weight_N": float(weight),
                "normal_total_N": float(normal_total),
                "required_total_normal_N": float(required_total),
                "required_per_finger_N": float(required_per_finger),
                "friction_capacity_N": float(capacity),
                "margin_N": float(capacity - weight),
                "margin_ratio": float(capacity / weight) if weight > 0.0 else float("inf"),
                "pass": bool(capacity >= weight),
            }

        def current_passive_joint_pos():
            joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
            out = []
            for idx in passive_log_ids:
                out.append(float(joint_pos[idx]) if idx is not None else float("nan"))
            return out

        def physics_trace_row(phase, target_value=None):
            snap = current_live_contact_snapshot()
            force = current_contact_forces()
            friction_margin = friction_margin_from_forces(force)
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
                "pen_stat_m": snap.get("pen_stat_m", snap["pen_m"]),
                "pen_max_m": snap.get("pen_max_m", snap["pen_m"]),
                "pen_p90_m": snap.get("pen_p90_m", {}),
                "pen_mean_contact_m": snap.get("pen_mean_contact_m", {}),
                "pen_contact_fraction": snap.get("pen_contact_fraction", {}),
                "force_N": {
                    side: float(force.get("forces", {}).get(side, {}).get("normal_N", 0.0))
                    for side in ("left", "right")
                },
                "force_world_N": {
                    side: force.get("forces", {}).get(side, {}).get("world_N", [0.0, 0.0, 0.0])
                    for side in ("left", "right")
                },
                "force_source": force.get("source", "none"),
                "friction_margin": friction_margin,
                "object_pose_w": snap["object_pose_w"],
                "pad_pose_w": snap["pad_pose_w"],
                "gel_origin_w": snap["gel_origin_w"],
                "gel_gap": snap["gel_gap"],
                "grip_symmetry": grip_symmetry_from_snapshot(snap),
                "object_pos_w": obj_pos,
                "robot_root_z_m": float(root_pos[2]),
            }

        def safe_grip_pen_limit(recess):
            safe_cap = max(0.0, 2.0 * float(recess) - 1.0e-5)
            return min(max(float(args.max_live_placement_pen), 0.0), safe_cap)

        def pen_max_explosion_limit():
            # The analytic map is clipped at 0.75 mm; touching that clip is
            # acceptable for tilted-corner artifacts, but anything above it is
            # treated as a pathological placement.
            return 7.6e-4

        def grip_pen_target(recess):
            requested = max(float(args.physics_grip_pen), float(recess) + 5.0e-5)
            return min(requested, safe_grip_pen_limit(recess))

        def assert_placement_within_recess(label, require_engaged=False):
            if not args.object_at_pads:
                return None
            snap = current_live_contact_snapshot()
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            limit = safe_grip_pen_limit(recess)
            pen_stats = snap.get("pen_stat_m", snap["pen_m"])
            pen_max = snap.get("pen_max_m", snap["pen_m"])
            over = {side: pen for side, pen in pen_stats.items() if pen >= limit}
            under = {
                side: pen
                for side, pen in pen_stats.items()
                if require_engaged and pen <= recess
            }
            explosion_limit = pen_max_explosion_limit()
            max_over = {side: pen for side, pen in pen_max.items() if pen > explosion_limit}
            if over or under or max_over:
                raise RuntimeError(
                    f"{label} placement outside safe grip penetration band: "
                    f"over_limit={over}, under_recess={under}, max_over={max_over}, "
                    f"limit={limit:.6g}, recess={recess:.6g}, "
                    f"pen_max_explosion_limit={explosion_limit:.6g}. "
                    "Use a smaller --physics-grip-pen or widen the calibrated close sweep."
                )
            return snap

        def assert_trace_pen_within_grip_band(
            label,
            rows,
            phases=("SQUEEZE", "HOLD", "LIFT"),
            raise_on_failure=True,
        ):
            if not rows:
                return None
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            limit = safe_grip_pen_limit(recess)
            checked = []
            over = []
            under = []
            phase_set = set(phases)
            for row in rows:
                if row.get("phase") not in phase_set:
                    continue
                if (
                    row.get("phase") == "SQUEEZE"
                    and int(row.get("step_in_phase") or 0) <= int(args.physics_squeeze_pen_check_after_steps)
                ):
                    continue
                pens = row.get("pen_stat_m") or row.get("pen_m", {})
                pen_max = row.get("pen_max_m") or row.get("pen_m", {})
                if not pens:
                    continue
                checked.append(row)
                for side, pen in pens.items():
                    pen = float(pen)
                    entry = {
                        "phase": row.get("phase"),
                        "step_in_phase": row.get("step_in_phase"),
                        "side": side,
                        "pen_m": pen,
                    }
                    if pen >= limit:
                        over.append(entry)
                    if pen <= recess:
                        under.append(entry)
                for side, pen in pen_max.items():
                    pen = float(pen)
                    if pen > pen_max_explosion_limit():
                        over.append({
                            "phase": row.get("phase"),
                            "step_in_phase": row.get("step_in_phase"),
                            "side": side,
                            "pen_max_m": pen,
                            "pen_max_explosion_limit_m": float(pen_max_explosion_limit()),
                        })
            result = {
                "phases": list(phases),
                "rows_checked": len(checked),
                "recess_m": float(recess),
                "limit_m": float(limit),
                "squeeze_pen_check_after_steps": int(args.physics_squeeze_pen_check_after_steps),
                "pass": bool(checked and not over and not under),
                "over_limit": over[:8],
                "under_recess": under[:8],
            }
            if (over or under) and raise_on_failure:
                raise RuntimeError(
                    f"{label} grip trace outside safe penetration band: "
                    f"over_limit={over[:8]}, under_recess={under[:8]}, "
                    f"limit={limit:.6g}, recess={recess:.6g}"
                )
            return result

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
                pen_vals = list(snap.get("pen_stat_m", snap["pen_m"]).values())
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
                    friction_margin = friction_margin_from_forces(force)
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
                        "friction_margin": friction_margin,
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

        def run_gripper_target_steps(
            target_value,
            steps,
            phase,
            trace_rows,
            root_pose_fn=None,
            start_value=None,
            ramp_steps=None,
            hold_object_pose=None,
            hold_object_steps=0,
            hold_release_pen=None,
            hold_live_contact_depth=None,
        ):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return []
            import torch

            def write_object_hold_pose():
                if hold_object_pose is None or not hasattr(obj, "write_root_pose_to_sim"):
                    return
                if hold_live_contact_depth is not None:
                    center_np = object_center_from_live_pads(contact_depth=float(hold_live_contact_depth))
                    center_np, _, _ = balance_live_contact_center(
                        center_np,
                        float(hold_live_contact_depth),
                    )
                    center_np = center_np + _vec3_arg(args.object_center_offset_world)
                    pose = obj.data.default_root_state.clone()
                    pose[0, 0:3] = (
                        torch.tensor(center_np, device=obj.device, dtype=torch.float32)
                        + torch.tensor([0.0, 0.0, args.object_pad_z_offset], device=obj.device, dtype=torch.float32)
                    )
                    pose[0, 3:7] = hold_object_pose[0, 3:7]
                    obj.write_root_pose_to_sim(pose[:, :7])
                else:
                    obj.write_root_pose_to_sim(hold_object_pose)
                if hasattr(obj, "write_root_velocity_to_sim"):
                    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=obj.device))

            phase_rows = []
            object_hold_released = hold_object_pose is None
            release_pen = hold_release_pen
            if release_pen is None or float(release_pen) < 0.0:
                release_pen = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            min_hold_steps = int(max(hold_object_steps, 0))
            for k in range(int(max(steps, 0))):
                if start_value is None:
                    step_target_value = float(target_value)
                else:
                    denom_steps = int(max(1, ramp_steps if ramp_steps is not None else steps))
                    alpha = min(1.0, (k + 1) / float(denom_steps))
                    step_target_value = (1.0 - alpha) * float(start_value) + alpha * float(target_value)
                object_hold_active = bool(not object_hold_released)
                if object_hold_active:
                    write_object_hold_pose()
                target = torch.full((1, len(gripper_ids)), float(step_target_value), device=robot.device)
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
                if object_hold_active:
                    write_object_hold_pose()
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)
                row = physics_trace_row(phase, step_target_value)
                row["step_in_phase"] = int(k + 1)
                row["object_hold_active"] = object_hold_active
                row["object_hold_live_recenter"] = bool(object_hold_active and hold_live_contact_depth is not None)
                row["object_hold_release_pen_m"] = float(release_pen)
                trace_rows.append(row)
                phase_rows.append(row)
                pens = row.get("pen_stat_m") or row.get("pen_m", {})
                pen_vals = [
                    float(pens.get(side, 0.0))
                    for side in ("left", "right")
                    if side in pens
                ]
                if (
                    object_hold_active
                    and (k + 1) >= min_hold_steps
                    and len(pen_vals) == 2
                    and min(pen_vals) > float(release_pen)
                ):
                    object_hold_released = True
                    row["object_hold_released_after_step"] = True
            return phase_rows

        def summarize_phase_rows(rows):
            if not rows:
                return {
                    "rows": 0,
                    "min_pen_L_m": 0.0,
                    "min_pen_R_m": 0.0,
                    "min_force_L_N": 0.0,
                    "min_force_R_N": 0.0,
                    "max_force_L_N": 0.0,
                    "max_force_R_N": 0.0,
                    "min_friction_margin_N": 0.0,
                    "min_friction_margin_ratio": 0.0,
                    "min_object_z_m": 0.0,
                    "max_object_z_m": 0.0,
                    "final_object_z_m": 0.0,
                    "force_spike_pass": True,
                }
            force_l = [float(r.get("force_N", {}).get("left", 0.0)) for r in rows]
            force_r = [float(r.get("force_N", {}).get("right", 0.0)) for r in rows]
            pen_l = [float((r.get("pen_stat_m") or r.get("pen_m", {})).get("left", 0.0)) for r in rows]
            pen_r = [float((r.get("pen_stat_m") or r.get("pen_m", {})).get("right", 0.0)) for r in rows]
            pen_max_l = [float((r.get("pen_max_m") or r.get("pen_m", {})).get("left", 0.0)) for r in rows]
            pen_max_r = [float((r.get("pen_max_m") or r.get("pen_m", {})).get("right", 0.0)) for r in rows]
            margins = [float(r.get("friction_margin", {}).get("margin_N", 0.0)) for r in rows]
            ratios = [float(r.get("friction_margin", {}).get("margin_ratio", 0.0)) for r in rows]
            obj_z = [
                float(r.get("object_pos_w", [0.0, 0.0, 0.0])[2])
                for r in rows
                if len(r.get("object_pos_w", [])) >= 3
            ]
            max_force = max(max(force_l) if force_l else 0.0, max(force_r) if force_r else 0.0)
            return {
                "rows": int(len(rows)),
                "min_pen_L_m": float(min(pen_l)) if pen_l else 0.0,
                "min_pen_R_m": float(min(pen_r)) if pen_r else 0.0,
                "max_pen_L_m": float(max(pen_l)) if pen_l else 0.0,
                "max_pen_R_m": float(max(pen_r)) if pen_r else 0.0,
                "min_pen_max_L_m": float(min(pen_max_l)) if pen_max_l else 0.0,
                "min_pen_max_R_m": float(min(pen_max_r)) if pen_max_r else 0.0,
                "max_pen_max_L_m": float(max(pen_max_l)) if pen_max_l else 0.0,
                "max_pen_max_R_m": float(max(pen_max_r)) if pen_max_r else 0.0,
                "min_force_L_N": float(min(force_l)) if force_l else 0.0,
                "min_force_R_N": float(min(force_r)) if force_r else 0.0,
                "max_force_L_N": float(max(force_l)) if force_l else 0.0,
                "max_force_R_N": float(max(force_r)) if force_r else 0.0,
                "min_friction_margin_N": float(min(margins)) if margins else 0.0,
                "min_friction_margin_ratio": float(min(ratios)) if ratios else 0.0,
                "max_friction_margin_ratio": float(max(ratios)) if ratios else 0.0,
                "min_object_z_m": float(min(obj_z)) if obj_z else 0.0,
                "max_object_z_m": float(max(obj_z)) if obj_z else 0.0,
                "final_object_z_m": float(obj_z[-1]) if obj_z else 0.0,
                "force_spike_limit_N": float(args.physics_force_spike_limit_N),
                "max_force_N": float(max_force),
                "force_spike_pass": bool(max_force < args.physics_force_spike_limit_N),
            }

        def calibrate_close_ramp(start_target, end_target, steps):
            if not gripper_ids or not hasattr(robot, "set_joint_position_target"):
                return {"enabled": False, "rows": []}
            import torch

            rows = []
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            grip_pen = grip_pen_target(recess)
            target_gap = float(max(args.object_size - 2.0 * grip_pen, 0.0))
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
            placement_recess_margin = 1.0e-5
            placement_touch_pen = max(recess + placement_recess_margin, 0.0)
            placement_cap = safe_grip_pen_limit(recess)
            placement_target_pen = min(max(grip_pen, placement_touch_pen), placement_cap)
            nominal_desired_gap = nominal_gap_for_contact_depth(placement_target_pen)
            for row in valid:
                support_required_gap, supports = support_required_gap_for_gel_frames(
                    row["frames"], placement_target_pen
                )
                required_gap = nominal_desired_gap
                prediction = placement_prediction_for_gel_frames(row["frames"], placement_target_pen)
                row["placement_cap_m"] = float(placement_cap)
                row["placement_recess_margin_m"] = float(placement_recess_margin)
                row["physics_grip_pen_m"] = float(grip_pen)
                row["placement_target_pen_m"] = float(placement_target_pen)
                row["placement_nominal_gap_m"] = float(nominal_desired_gap)
                row["placement_required_gap_m"] = float(required_gap)
                row["placement_required_gap_source"] = "nominal_symmetric_object_size_minus_2pen"
                row["placement_support_m"] = supports
                row["placement_support_required_gap_m"] = (
                    None if support_required_gap is None else float(support_required_gap)
                )
                row["placement_gap_clearance_m"] = float(row["gap_m"] - required_gap)
                row["placement_desired_gap_m"] = float(required_gap)
                row["placement_desired_gap_error_m"] = float(abs(row["gap_m"] - required_gap))
                if prediction is not None:
                    row["placement_predicted_center"] = prediction["center"]
                    row["placement_predicted_base_center"] = prediction["base_center"]
                    row["placement_predicted_center_balance"] = prediction["center_balance"]
                    row["placement_predicted_pen_m"] = prediction["pen_m"]
                    row["placement_predicted_pen_stat_m"] = prediction["pen_stat_m"]
                    row["placement_predicted_pen_max_m"] = prediction["pen_max_m"]
                    row["placement_predicted_pen_stats"] = prediction["pen_stats"]
                    row["placement_predicted_min_pen_m"] = prediction["min_pen_m"]
                    row["placement_predicted_max_pen_m"] = prediction["max_pen_m"]
                    row["placement_predicted_mean_pen_m"] = prediction["mean_pen_m"]
                    row["placement_predicted_pen_error_m"] = prediction["pen_error_m"]
                    row["placement_predicted_mean_pen_delta_m"] = prediction["mean_pen_delta_m"]
                    row["placement_predicted_imbalance_m"] = prediction["imbalance_m"]
                    row["placement_pen_guard_pass"] = bool(
                        prediction["min_pen_m"] > recess
                        and prediction["max_pen_m"] < placement_cap
                        and prediction["imbalance_m"] <= placement_balance_tolerance(placement_target_pen)
                        and (
                            not prediction["pen_max_m"]
                            or max(prediction["pen_max_m"].values()) <= pen_max_explosion_limit()
                        )
                    )
                else:
                    row["placement_predicted_center"] = None
                    row["placement_predicted_base_center"] = None
                    row["placement_predicted_center_balance"] = None
                    row["placement_predicted_pen_m"] = None
                    row["placement_predicted_pen_stat_m"] = None
                    row["placement_predicted_pen_max_m"] = None
                    row["placement_predicted_pen_stats"] = None
                    row["placement_predicted_min_pen_m"] = None
                    row["placement_predicted_max_pen_m"] = None
                    row["placement_predicted_mean_pen_m"] = None
                    row["placement_predicted_pen_error_m"] = None
                    row["placement_predicted_mean_pen_delta_m"] = None
                    row["placement_predicted_imbalance_m"] = None
                    row["placement_pen_guard_pass"] = False
            bracket = None
            if placement_target_pen > 0.0:
                if valid and abs(float(valid[0]["placement_gap_clearance_m"])) <= 1.0e-12:
                    bracket = (valid[0], valid[0])
                for prev, curr in zip(valid, valid[1:]):
                    if bracket is not None:
                        break
                    prev_clearance = float(prev["placement_gap_clearance_m"])
                    curr_clearance = float(curr["placement_gap_clearance_m"])
                    if abs(curr_clearance) <= 1.0e-12 or prev_clearance * curr_clearance < 0.0:
                        bracket = (prev, curr)
                        break
            if bracket is not None:
                prev, curr = bracket
                prev_gap = float(prev["gap_m"])
                curr_gap = float(curr["gap_m"])
                prev_clearance = float(prev["placement_gap_clearance_m"])
                curr_clearance = float(curr["placement_gap_clearance_m"])
                denom = curr_clearance - prev_clearance
                frac = 0.0 if abs(denom) <= 1.0e-12 else -prev_clearance / denom
                frac = float(min(1.0, max(0.0, frac)))
                interp_target = (1.0 - frac) * float(prev["target"]) + frac * float(curr["target"])
                interp_step = (1.0 - frac) * float(prev["step"]) + frac * float(curr["step"])
                interp_gap = (1.0 - frac) * prev_gap + frac * curr_gap
                prev_required = float(prev["placement_required_gap_m"])
                curr_required = float(curr["placement_required_gap_m"])
                interp_required = (1.0 - frac) * prev_required + frac * curr_required
                prev_support_required = prev.get("placement_support_required_gap_m")
                curr_support_required = curr.get("placement_support_required_gap_m")
                interp_support_required = None
                if prev_support_required is not None and curr_support_required is not None:
                    interp_support_required = (
                        (1.0 - frac) * float(prev_support_required)
                        + frac * float(curr_support_required)
                    )

                def interp_optional(key):
                    prev_value = prev.get(key)
                    curr_value = curr.get(key)
                    if prev_value is None or curr_value is None:
                        return None
                    return (1.0 - frac) * float(prev_value) + frac * float(curr_value)

                interp_pen = interp_optional("placement_predicted_mean_pen_m")
                interp_min_pen = interp_optional("placement_predicted_min_pen_m")
                interp_max_pen = interp_optional("placement_predicted_max_pen_m")
                interp_imbalance = interp_optional("placement_predicted_imbalance_m")
                interp_pen_error = None
                interp_guard_pass = False
                if interp_min_pen is not None and interp_max_pen is not None:
                    interp_pen_error = max(
                        abs(interp_min_pen - placement_target_pen),
                        abs(interp_max_pen - placement_target_pen),
                    )
                    interp_guard_pass = bool(
                        interp_min_pen > recess
                        and interp_max_pen < placement_cap
                        and (
                            interp_imbalance is None
                            or interp_imbalance <= placement_balance_tolerance(placement_target_pen)
                        )
                    )
                place_row = {
                    "step": float(interp_step),
                    "target": float(interp_target),
                    "finger_target": float(interp_target),
                    "finger_joint_pos": None,
                    "gap_m": float(interp_gap),
                    "midpoint": None,
                    "frames": None,
                    "target_gap_error_m": float(abs(interp_gap - target_gap)),
                    "preclose_gap_error_m": float(abs(interp_gap - preclose_gap)),
                    "place_depth_mode": "interpolated_nominal_gap",
                    "placement_depth_m": float(placement_target_pen),
                    "placement_target_pen_m": float(placement_target_pen),
                    "placement_nominal_gap_m": float(nominal_desired_gap),
                    "placement_required_gap_m": float(interp_required),
                    "placement_required_gap_source": "nominal_symmetric_object_size_minus_2pen",
                    "placement_support_required_gap_m": (
                        None if interp_support_required is None else float(interp_support_required)
                    ),
                    "placement_gap_clearance_m": float(interp_gap - interp_required),
                    "placement_desired_gap_m": float(interp_required),
                    "placement_cap_m": float(placement_cap),
                    "placement_recess_margin_m": float(placement_recess_margin),
                    "physics_grip_pen_m": float(grip_pen),
                    "placement_predicted_mean_pen_m": None if interp_pen is None else float(interp_pen),
                    "placement_predicted_min_pen_m": None if interp_min_pen is None else float(interp_min_pen),
                    "placement_predicted_max_pen_m": None if interp_max_pen is None else float(interp_max_pen),
                    "placement_predicted_pen_error_m": None if interp_pen_error is None else float(interp_pen_error),
                    "placement_predicted_mean_pen_delta_m": (
                        None if interp_pen is None else float(interp_pen - placement_target_pen)
                    ),
                    "placement_predicted_imbalance_m": (
                        None if interp_imbalance is None else float(interp_imbalance)
                    ),
                    "placement_pen_guard_pass": bool(interp_guard_pass),
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
                            "placement_required_gap_m": prev.get("placement_required_gap_m"),
                            "placement_gap_clearance_m": prev.get("placement_gap_clearance_m"),
                            "placement_support_m": prev.get("placement_support_m"),
                            "placement_support_required_gap_m": prev.get("placement_support_required_gap_m"),
                            "placement_predicted_pen_m": prev.get("placement_predicted_pen_m"),
                            "placement_predicted_pen_error_m": prev.get("placement_predicted_pen_error_m"),
                            "placement_predicted_imbalance_m": prev.get("placement_predicted_imbalance_m"),
                            "placement_pen_guard_pass": prev.get("placement_pen_guard_pass"),
                        },
                        {
                            "step": int(curr["step"]),
                            "target": float(curr["target"]),
                            "finger_target": float(curr["target"]),
                            "gap_m": float(curr_gap),
                            "origin_gap_m": curr.get("origin_gap_m"),
                            "left_face_gap_m": curr.get("left_face_gap_m"),
                            "right_face_gap_m": curr.get("right_face_gap_m"),
                            "placement_required_gap_m": curr.get("placement_required_gap_m"),
                            "placement_gap_clearance_m": curr.get("placement_gap_clearance_m"),
                            "placement_support_m": curr.get("placement_support_m"),
                            "placement_support_required_gap_m": curr.get("placement_support_required_gap_m"),
                            "placement_predicted_pen_m": curr.get("placement_predicted_pen_m"),
                            "placement_predicted_pen_error_m": curr.get("placement_predicted_pen_error_m"),
                            "placement_predicted_imbalance_m": curr.get("placement_predicted_imbalance_m"),
                            "placement_pen_guard_pass": curr.get("placement_pen_guard_pass"),
                        },
                    ],
                }
                preclose_row = place_row
            if place_row is not None:
                place_row["place_center"] = None
                place_row["predicted_pen_m"] = None
                preclose_row = place_row
            gap_values = [float(r["gap_m"]) for r in valid]
            required_gap_values = [
                float(r["placement_required_gap_m"])
                for r in valid
                if r.get("placement_required_gap_m") is not None
            ]
            support_required_gap_values = [
                float(r["placement_support_required_gap_m"])
                for r in valid
                if r.get("placement_support_required_gap_m") is not None
            ]
            clearance_values = [
                float(r["placement_gap_clearance_m"])
                for r in valid
                if r.get("placement_gap_clearance_m") is not None
            ]
            predicted_pen_errors = [
                float(r["placement_predicted_pen_error_m"])
                for r in valid
                if r.get("placement_predicted_pen_error_m") is not None
            ]
            predicted_max_pens = [
                float(r["placement_predicted_max_pen_m"])
                for r in valid
                if r.get("placement_predicted_max_pen_m") is not None
            ]
            predicted_guard_rows = [r for r in valid if r.get("placement_pen_guard_pass")]
            origin_gap_values = [
                float(r["origin_gap_m"])
                for r in valid
                if r.get("origin_gap_m") is not None
            ]
            failure_detail = None
            if place_row is None:
                if gap_values:
                    failure_detail = (
                        "no calibrated close row bracket crossed nominal symmetric placement clearance "
                        f"for nominal_gap={nominal_desired_gap:.6g} m target_pen={placement_target_pen:.6g} m "
                        f"within cap={placement_cap:.6g} m; "
                        f"rows={len(rows)} valid_rows={len(valid)} "
                        f"first_gap={gap_values[0]:.6g} m last_gap={gap_values[-1]:.6g} m "
                        f"min_gap={min(gap_values):.6g} m max_gap={max(gap_values):.6g} m"
                    )
                    if required_gap_values:
                        failure_detail += (
                            f" min_required_gap={min(required_gap_values):.6g} m "
                            f"max_required_gap={max(required_gap_values):.6g} m"
                        )
                    if support_required_gap_values:
                        failure_detail += (
                            f" min_support_required_gap={min(support_required_gap_values):.6g} m "
                            f"max_support_required_gap={max(support_required_gap_values):.6g} m"
                        )
                    if clearance_values:
                        failure_detail += (
                            f" first_clearance={clearance_values[0]:.6g} m "
                            f"last_clearance={clearance_values[-1]:.6g} m "
                            f"min_clearance={min(clearance_values):.6g} m "
                            f"max_clearance={max(clearance_values):.6g} m"
                        )
                    if predicted_pen_errors:
                        failure_detail += (
                            f" min_predicted_pen_error={min(predicted_pen_errors):.6g} m "
                            f"min_predicted_max_pen={min(predicted_max_pens):.6g} m "
                            f"max_predicted_max_pen={max(predicted_max_pens):.6g} m "
                            f"guarded_predicted_rows={len(predicted_guard_rows)}"
                        )
                    if origin_gap_values:
                        failure_detail += (
                            f" first_origin_gap={origin_gap_values[0]:.6g} m "
                            f"last_origin_gap={origin_gap_values[-1]:.6g} m"
                        )
                else:
                    failure_detail = (
                        "no calibrated close row bracket crossed nominal symmetric placement clearance "
                        f"for nominal_gap={nominal_desired_gap:.6g} m target_pen={placement_target_pen:.6g} m "
                        f"within cap={placement_cap:.6g} m; rows={len(rows)} valid_rows=0"
                    )
            return {
                "enabled": True,
                "target_gap_m": target_gap,
                "preclose_gap_m": preclose_gap,
                "physics_grip_pen_m": float(grip_pen),
                "placement_cap_m": float(placement_cap),
                "placement_recess_margin_m": float(placement_recess_margin),
                "placement_target_pen_m": float(placement_target_pen),
                "placement_nominal_gap_m": float(nominal_desired_gap),
                "placement_desired_gap_m": None if place_row is None else place_row.get("placement_desired_gap_m"),
                "place_center": None if place_center is None else place_center.tolist(),
                "place_row": place_row,
                "preclose_row": preclose_row,
                "gap_metric": "mean signed distance between gel face planes along inward pad normals",
                "origin_gap_metric": "Euclidean distance between gel frame origins",
                "placement_failure": failure_detail,
                "rows": rows,
            }

        def calibration_gap_slope(rows, target_value):
            points = sorted(
                (
                    (float(r["target"]), float(r["gap_m"]), int(r.get("step", 0)))
                    for r in rows
                    if r.get("gap_m") is not None
                    and np.isfinite(float(r["target"]))
                    and np.isfinite(float(r["gap_m"]))
                ),
                key=lambda x: x[0],
            )
            if len(points) < 2:
                return None
            target_value = float(target_value)
            pair = None
            for left, right in zip(points, points[1:]):
                if left[0] <= target_value <= right[0] or right[0] <= target_value <= left[0]:
                    if abs(right[0] - left[0]) > 1.0e-12:
                        pair = (left, right)
                        break
            if pair is None:
                nearest = sorted(points, key=lambda x: abs(x[0] - target_value))
                for i, left in enumerate(nearest):
                    for right in nearest[i + 1:]:
                        if abs(right[0] - left[0]) > 1.0e-12:
                            pair = (left, right)
                            break
                    if pair is not None:
                        break
            if pair is None:
                return None
            left, right = pair
            slope = (right[1] - left[1]) / (right[0] - left[0])
            if not np.isfinite(slope) or abs(slope) < 1.0e-9:
                return None
            return {
                "slope_gap_per_target": float(slope),
                "points": [
                    {"target": float(left[0]), "gap_m": float(left[1]), "step": int(left[2])},
                    {"target": float(right[0]), "gap_m": float(right[1]), "step": int(right[2])},
                ],
            }

        def refine_calibrated_close_target(calib, initial_target, contact_depth):
            rows = calib.get("rows", [])
            valid_targets = [
                float(r["target"])
                for r in rows
                if r.get("gap_m") is not None and np.isfinite(float(r["target"]))
            ]
            if not valid_targets:
                raise RuntimeError("calibrated close live gap refinement has no valid dry-run target rows")
            target_min = min(valid_targets)
            target_max = max(valid_targets)
            target_value = float(initial_target)
            placement_recess_margin = float(calib.get("placement_recess_margin_m", 5.0e-5))
            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            placement_touch_pen = max(recess + placement_recess_margin, 0.0)
            placement_safe_pen = safe_grip_pen_limit(recess)
            object_fit_clearance = 1.0e-4
            # Live settled gap is noisy/non-monotonic at sub-0.1 mm scale; a
            # +/-0.3 mm band is enough for clearance placement before close.
            band_half_width = 3.0e-4
            pen_accept_tolerance = 7.5e-5
            max_updates = 4
            settle_steps = max(4, int(args.physics_preclose_steps))
            iterations = []
            success = False
            final_state = None
            best = None

            def evaluate_live_gap(state):
                support_required_gap, live_supports = support_required_gap_for_gel_frames(
                    state["frames"], contact_depth
                )
                live_required_gap, _ = required_gap_for_gel_frames(state["frames"], contact_depth)
                desired_gap = live_required_gap
                if desired_gap is None:
                    place_row = calib.get("place_row") or {}
                    desired_gap = place_row.get("placement_desired_gap_m")
                measured_gap = state.get("gap_m")
                nominal_gap_error = None if measured_gap is None or desired_gap is None else float(measured_gap - desired_gap)
                error = nominal_gap_error
                placement_min_gap = max(float(args.object_size) - 2.0 * placement_safe_pen, 0.0)
                nominal_max_gap = max(float(args.object_size) - 2.0 * placement_touch_pen, 0.0)
                band_min_gap = None if desired_gap is None else float(desired_gap) - band_half_width
                band_max_gap = None if desired_gap is None else float(desired_gap) + band_half_width
                prospective_pen = None
                finite_measured = measured_gap is not None and np.isfinite(float(measured_gap))
                if finite_measured:
                    prospective_pen = max(0.5 * (float(args.object_size) - float(measured_gap)), 0.0)
                prediction = placement_prediction_for_gel_frames(state["frames"], contact_depth)
                predicted_min_pen = None if prediction is None else prediction["min_pen_m"]
                predicted_max_pen = None if prediction is None else prediction["max_pen_m"]
                predicted_mean_pen = None if prediction is None else prediction["mean_pen_m"]
                predicted_pen = None if prediction is None else prediction["pen_m"]
                predicted_pen_stat = None if prediction is None else prediction["pen_stat_m"]
                predicted_pen_max = None if prediction is None else prediction["pen_max_m"]
                predicted_pen_stats = None if prediction is None else prediction["pen_stats"]
                predicted_pen_error = None if prediction is None else prediction["pen_error_m"]
                predicted_pen_delta = None if prediction is None else prediction["mean_pen_delta_m"]
                predicted_imbalance = None if prediction is None else prediction["imbalance_m"]
                predicted_center_balance = None if prediction is None else prediction["center_balance"]
                if predicted_pen_delta is not None and np.isfinite(float(predicted_pen_delta)):
                    # Convert the analytic mean-penetration error into the
                    # same sign convention as measured_gap - desired_gap:
                    # positive means the jaws are too open, negative too deep.
                    error = float(-2.0 * float(predicted_pen_delta))
                finite_error = error is not None and np.isfinite(float(error))
                object_fits = finite_measured and float(measured_gap) >= placement_min_gap
                within_recess_reach = (
                    predicted_min_pen is not None
                    and float(predicted_min_pen) > recess
                )
                within_recess_guard = (
                    predicted_max_pen is not None
                    and float(predicted_max_pen) < placement_safe_pen
                    and (
                        not predicted_pen_max
                        or max(predicted_pen_max.values()) <= pen_max_explosion_limit()
                    )
                )
                within_band = finite_error and abs(error) <= band_half_width
                within_pen_target = (
                    predicted_pen_error is not None
                    and float(predicted_pen_error) <= pen_accept_tolerance
                )
                within_pen_balance = (
                    predicted_imbalance is not None
                    and float(predicted_imbalance) <= placement_balance_tolerance(contact_depth)
                )
                valid_placement = bool(
                    object_fits
                    and within_band
                    and within_recess_reach
                    and within_recess_guard
                    and within_pen_balance
                )
                return {
                    "desired_gap_m": desired_gap,
                    "measured_gap_m": measured_gap,
                    "gap_error_m": error,
                    "nominal_gap_error_m": nominal_gap_error,
                    "support_m": live_supports,
                    "support_required_gap_m": support_required_gap,
                    "required_gap_source": "nominal_symmetric_object_size_minus_2pen",
                    "placement_band_half_width_m": float(band_half_width),
                    "placement_band_min_gap_m": band_min_gap,
                    "placement_band_max_gap_m": band_max_gap,
                    "placement_min_gap_m": float(placement_min_gap),
                    "placement_max_gap_m": float(nominal_max_gap),
                    "placement_nominal_recess_max_gap_m": float(nominal_max_gap),
                    "placement_object_fit_clearance_m": float(object_fit_clearance),
                    "placement_recess_guard_pen_m": float(placement_safe_pen),
                    "placement_touch_pen_m": float(placement_touch_pen),
                    "prospective_per_side_pen_m": prospective_pen,
                    "predicted_pen_m": predicted_pen,
                    "predicted_pen_stat_m": predicted_pen_stat,
                    "predicted_pen_max_m": predicted_pen_max,
                    "predicted_pen_stats": predicted_pen_stats,
                    "predicted_min_pen_m": predicted_min_pen,
                    "predicted_max_pen_m": predicted_max_pen,
                    "predicted_mean_pen_m": predicted_mean_pen,
                    "predicted_pen_error_m": predicted_pen_error,
                    "predicted_mean_pen_delta_m": predicted_pen_delta,
                    "predicted_imbalance_m": predicted_imbalance,
                    "predicted_center_balance": predicted_center_balance,
                    "placement_pen_accept_tolerance_m": float(pen_accept_tolerance),
                    "placement_balance_tolerance_m": float(placement_balance_tolerance(contact_depth)),
                    "object_fits": bool(object_fits),
                    "within_recess_reach": bool(within_recess_reach),
                    "within_recess_guard": bool(within_recess_guard),
                    "within_pen_balance": bool(within_pen_balance),
                    "within_band": bool(within_band),
                    "within_pen_target": bool(within_pen_target),
                    "valid_placement": bool(valid_placement),
                    "accepted": bool(valid_placement),
                }

            def live_error_bracket():
                finite = [
                    r for r in iterations
                    if r.get("gap_error_m") is not None and np.isfinite(float(r["gap_error_m"]))
                ]
                bracket = None
                for left in finite:
                    for right in finite:
                        if float(left["target"]) == float(right["target"]):
                            continue
                        if float(left["gap_error_m"]) * float(right["gap_error_m"]) <= 0.0:
                            span = abs(float(right["target"]) - float(left["target"]))
                            if bracket is None or span < bracket[0]:
                                bracket = (span, left, right)
                return None if bracket is None else (bracket[1], bracket[2])

            for i in range(max_updates + 1):
                state = gel_gap_state()
                final_state = state
                assessment = evaluate_live_gap(state)
                error = assessment["gap_error_m"]
                iter_row = {
                    "iteration": int(i),
                    "target": float(target_value),
                    "frames": state.get("frames"),
                    "normal_dot": state.get("normal_dot"),
                    "left_face_gap_m": state.get("left_face_gap_m"),
                    "right_face_gap_m": state.get("right_face_gap_m"),
                }
                iter_row.update(assessment)
                iterations.append(iter_row)
                pen_error = iter_row.get("predicted_pen_error_m")
                if iter_row.get("valid_placement"):
                    if pen_error is not None and np.isfinite(float(pen_error)):
                        if (
                            best is None
                            or best.get("predicted_pen_error_m") is None
                            or abs(float(pen_error)) < abs(float(best["predicted_pen_error_m"]))
                        ):
                            best = iter_row
                    elif best is None:
                        best = iter_row
                if iter_row["accepted"]:
                    success = True
                    break
                if i >= max_updates:
                    break
                slope_info = calibration_gap_slope(rows, target_value)
                iter_row["slope"] = slope_info
                if slope_info is None or error is None or not np.isfinite(float(error)):
                    break
                bracket = live_error_bracket()
                if bracket is not None:
                    left, right = bracket
                    next_target = 0.5 * (float(left["target"]) + float(right["target"]))
                    iter_row["update_mode"] = "live_bisection"
                    iter_row["live_bracket"] = [
                        {"target": float(left["target"]), "gap_error_m": float(left["gap_error_m"])},
                        {"target": float(right["target"]), "gap_error_m": float(right["gap_error_m"])},
                    ]
                else:
                    raw_next = target_value - error / float(slope_info["slope_gap_per_target"])
                    next_target = target_value + 0.5 * (raw_next - target_value)
                    iter_row["update_mode"] = "damped_dry_run_secant"
                    iter_row["raw_next_target"] = float(raw_next)
                next_target = float(min(target_max, max(target_min, next_target)))
                iter_row["next_target"] = next_target
                if abs(next_target - target_value) <= 1.0e-7:
                    break
                target_value = next_target
                set_gripper(target_value, settle_steps)
            used_best_fallback = False
            if (
                not success
                and best is not None
                and best.get("valid_placement")
            ):
                used_best_fallback = True
                success = True
                target_value = float(best["target"])
                if final_state is None or abs(target_value - float(iterations[-1]["target"])) > 1.0e-7:
                    set_gripper(target_value, settle_steps)
                    final_state = gel_gap_state()
            result = {
                "enabled": True,
                "initial_target": float(initial_target),
                "final_target": float(target_value),
                "acceptance": "valid placement inside object-fit, live-gap band, robust recess/guard, and balance checks; predicted pen target error is ranking-only",
                "placement_band_half_width_m": float(band_half_width),
                "placement_pen_accept_tolerance_m": float(pen_accept_tolerance),
                "placement_object_fit_clearance_m": float(object_fit_clearance),
                "placement_recess_guard_pen_m": float(placement_safe_pen),
                "placement_touch_pen_m": float(placement_touch_pen),
                "settle_steps": int(settle_steps),
                "target_bounds": [float(target_min), float(target_max)],
                "success": bool(success),
                "used_best_fallback": bool(used_best_fallback),
                "best_target": None if best is None else float(best["target"]),
                "best_gap_error_m": None if best is None else float(best["gap_error_m"]),
                "best_predicted_pen_error_m": None if best is None else best.get("predicted_pen_error_m"),
                "best_valid_placement": bool(best.get("valid_placement")) if best is not None else False,
                "iterations": iterations,
            }
            calib["live_gap_refinement"] = result
            if not success:
                measured = [
                    r.get("measured_gap_m")
                    for r in iterations
                    if r.get("measured_gap_m") is not None
                ]
                desired = [
                    r.get("desired_gap_m")
                    for r in iterations
                    if r.get("desired_gap_m") is not None
                ]
                targets = [r.get("target") for r in iterations]
                validity = [
                    {
                        "target": r.get("target"),
                        "gap_error_m": r.get("gap_error_m"),
                        "within_band": r.get("within_band"),
                        "within_pen_target": r.get("within_pen_target"),
                        "predicted_pen_m": r.get("predicted_pen_m"),
                        "predicted_pen_error_m": r.get("predicted_pen_error_m"),
                        "valid_placement": r.get("valid_placement"),
                        "object_fits": r.get("object_fits"),
                        "within_recess_reach": r.get("within_recess_reach"),
                        "within_recess_guard": r.get("within_recess_guard"),
                    }
                    for r in iterations
                ]
                raise RuntimeError(
                    "calibrated close live gap refinement failed: "
                    f"targets={targets} measured_gap_m={measured} "
                    f"desired_gap_m={desired} band_half_width_m={band_half_width:.6g} "
                    f"pen_accept_tolerance_m={pen_accept_tolerance:.6g} "
                    f"object_fit_clearance_m={object_fit_clearance:.6g} "
                    f"recess_guard_pen_m={placement_safe_pen:.6g} validity={validity}"
                )
            return target_value, final_state

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
            supports = []
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
                support = _support_extent_for_object(args.object, args.object_size, T_w_gel[:3, 2])
                desired = support - depth
                centers.append(T_w_gel @ np.array([0.0, 0.0, desired, 1.0]))
                origins.append(T_w_gel[:3, 3].copy())
                normals.append(T_w_gel[:3, 2].copy())
                supports.append(support)
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
            if args.object_center_solver == "lsq" and not has_tangent_constraint:
                balanced = object_center_for_opposing_normals(origins, normals, supports)
                if balanced is not None:
                    return balanced
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

        def placement_center_diagnostics(center_np, contact_depth=None):
            center = np.asarray(center_np, dtype=np.float64)
            frames = current_gel_frames_w()
            snap = {
                "object_pose_w": {"pos": center.tolist()},
                "gel_origin_w": {
                    side: frame["origin"]
                    for side, frame in frames.items()
                    if isinstance(frame, dict) and frame.get("origin") is not None
                },
                "gel_frame_w": frames,
                "pad_pose_w": current_pad_pose_w(),
            }
            symmetry = grip_symmetry_from_snapshot(snap)
            prediction = predicted_contact_for_gel_frames(center, frames)
            return {
                "center": center.tolist(),
                "contact_depth_m": None if contact_depth is None else float(contact_depth),
                "gel_gap": gel_gap_from_frames(frames),
                "predicted_pen_m": prediction,
                "grip_symmetry": symmetry,
            }

        def balance_live_contact_center(center_np, contact_depth, state=None):
            state = gel_gap_state() if state is None else state
            balanced, balance = balance_center_for_gel_frames(center_np, state["frames"], contact_depth)
            return balanced, state, balance

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
            z = center_np[2] - bottom_off - scene_pedestal_height / 2.0
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
                "top_z": float(z + scene_pedestal_height / 2.0),
                "height": float(scene_pedestal_height),
                "height_source": scene_pedestal_height_source,
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
        physics_place_contact_depth = None
        if args.physics_lift_probe and args.grasp_mode == "ik" and gripper_ids:
            import torch
            from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
            from isaaclab.utils.math import subtract_frame_transforms

            physics_probe = {
                "enabled": True,
                "grasp_mode": "ik",
                "state_machine": [
                    "HOME",
                    "PRE_GRASP",
                    "DESCEND",
                    "ALIGN",
                    "CLOSE",
                    "SETTLE",
                    "LIFT",
                    "HOLD",
                ],
            }
            physics_trace = []
            physics_probe["trace"] = physics_trace
            physics_probe["ik_setup"] = {
                "arm_ready_joints": _float_list_arg(args.ik_arm_ready_joints, 6),
                "dls_damping": float(args.ik_dls_damping),
                "solver": str(args.ik_solver),
                "orientation_weight": float(args.ik_orientation_weight),
                "lift_orientation_weight": (
                    None if float(args.ik_lift_orientation_weight) < 0.0
                    else float(args.ik_lift_orientation_weight)
                ),
                "hold_orientation_weight": (
                    None if float(args.ik_hold_orientation_weight) < 0.0
                    else float(args.ik_hold_orientation_weight)
                ),
                "control_substeps": int(args.ik_control_substeps),
                "max_joint_step_rad": float(args.ik_max_joint_step),
                "diagnostics_log_every": int(args.ik_diagnostics_log_every),
                "pedestal_height_m": float(scene_pedestal_height),
                "pedestal_height_source": scene_pedestal_height_source,
                "min_close_ee_z_m": None if float(args.ik_min_close_ee_z) <= 0.0 else float(args.ik_min_close_ee_z),
                "soft_close": {
                    "effective_close_steps": int(max(args.physics_close_steps, args.ik_soft_close_steps)),
                    "configured_physics_close_steps": int(args.physics_close_steps),
                    "ik_soft_close_steps": int(args.ik_soft_close_steps),
                    "gripper_stiffness": float(effective_gripper_stiffness),
                    "gripper_damping": float(effective_gripper_damping),
                    "gripper_effort_limit": float(effective_gripper_effort_limit),
                },
                "phase_gripper_control": {
                    "lift": {
                        "stiffness": None if float(args.ik_lift_gripper_stiffness) < 0.0 else float(args.ik_lift_gripper_stiffness),
                        "damping": None if float(args.ik_lift_gripper_damping) < 0.0 else float(args.ik_lift_gripper_damping),
                        "effort_limit": None if float(args.ik_lift_gripper_effort_limit) < 0.0 else float(args.ik_lift_gripper_effort_limit),
                    },
                    "hold": {
                        "stiffness": None if float(args.ik_hold_gripper_stiffness) < 0.0 else float(args.ik_hold_gripper_stiffness),
                        "damping": None if float(args.ik_hold_gripper_damping) < 0.0 else float(args.ik_hold_gripper_damping),
                        "effort_limit": None if float(args.ik_hold_gripper_effort_limit) < 0.0 else float(args.ik_hold_gripper_effort_limit),
                    },
                },
            }
            if not arm_ids:
                raise RuntimeError("IK grasp mode requires UR arm joints")
            ee_matches = [i for i, name in enumerate(body_names) if name == args.ik_ee_body_name]
            if not ee_matches:
                ee_matches = [i for i, name in enumerate(body_names) if "wrist_3" in name]
            if not ee_matches:
                raise RuntimeError(
                    f"IK grasp mode could not find ee body {args.ik_ee_body_name!r}; "
                    f"available bodies={body_names}"
                )
            ee_body_id = int(ee_matches[0])
            diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": float(args.ik_dls_damping)},
            )
            diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=robot.device)

            def tensor_pose_w(pos_np, quat_np):
                pos = torch.tensor(np.asarray(pos_np, dtype=np.float32), device=robot.device).reshape(1, 3)
                quat = torch.tensor(np.asarray(quat_np, dtype=np.float32), device=robot.device).reshape(1, 4)
                return pos, quat

            def current_ee_pose_w():
                return (
                    robot.data.body_pos_w[:, ee_body_id].detach().clone(),
                    robot.data.body_quat_w[:, ee_body_id].detach().clone(),
                )

            def ee_target_command_b(pos_w_np, quat_w_tensor):
                target_pos_w, _ = tensor_pose_w(pos_w_np, quat_w_tensor.detach().cpu().numpy()[0])
                target_quat_w = quat_w_tensor.detach().clone()
                return torch.cat(
                    subtract_frame_transforms(
                        robot.data.root_pos_w,
                        robot.data.root_quat_w,
                        target_pos_w,
                        target_quat_w,
                    ),
                    dim=-1,
                )

            def current_ee_pose_b():
                ee_pos_w, ee_quat_w = current_ee_pose_w()
                return subtract_frame_transforms(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    ee_pos_w,
                    ee_quat_w,
                )

            def current_ee_jacobian():
                jac = robot.root_physx_view.get_jacobians()
                if not torch.is_tensor(jac):
                    jac = torch.as_tensor(jac, device=robot.device, dtype=torch.float32)
                jac_body_id = ee_body_id
                if jac_body_id >= jac.shape[1]:
                    jac_body_id = max(0, ee_body_id - 1)
                if jac.shape[1] == max(len(body_names) - 1, 1) and ee_body_id > 0:
                    jac_body_id = min(jac.shape[1] - 1, ee_body_id - 1)
                return jac[:, jac_body_id, :, arm_ids]

            def jacobian_diagnostics(jac):
                try:
                    jac_cpu = jac.detach().cpu().to(dtype=torch.float64)
                    jjt = jac_cpu @ jac_cpu.transpose(-1, -2)
                    eye = torch.eye(jjt.shape[-1], dtype=jjt.dtype).reshape(1, jjt.shape[-1], jjt.shape[-1])
                    damping = float(args.ik_dls_damping)
                    damped = jjt + (damping * damping) * eye
                    singular_values = torch.linalg.svdvals(jac_cpu[0]) if jac_cpu.numel() else torch.zeros(0, dtype=jac_cpu.dtype)
                    finite_sv = singular_values[singular_values > 1.0e-12]
                    condition = (
                        float((finite_sv.max() / finite_sv.min()).item())
                        if finite_sv.numel() > 0
                        else None
                    )
                    return {
                        "shape": list(jac_cpu.shape),
                        "rank": int(torch.linalg.matrix_rank(jac_cpu[0]).item()) if jac_cpu.numel() else 0,
                        "singular_values": singular_values.tolist(),
                        "condition": condition,
                        "jjt_diag": jjt[0].diag().tolist() if jjt.numel() else [],
                        "jjt_det": float(torch.linalg.det(jjt[0]).item()) if jjt.numel() else None,
                        "jjt_singular_values": torch.linalg.svdvals(jjt[0]).tolist() if jjt.numel() else [],
                        "dls_damping": damping,
                        "damped_jjt_det": float(torch.linalg.det(damped[0]).item()) if damped.numel() else None,
                        "damped_jjt_diag": damped[0].diag().tolist() if damped.numel() else [],
                    }
                except Exception as diag_exc:
                    return {"diagnostic_error": f"{type(diag_exc).__name__}: {diag_exc}"}

            def clamp_joint_delta(arm_joint_pos, arm_joint_des):
                max_step = float(args.ik_max_joint_step)
                raw_delta = arm_joint_des - arm_joint_pos
                if max_step > 0.0:
                    delta = torch.clamp(raw_delta, min=-max_step, max=max_step)
                    return arm_joint_pos + delta, raw_delta, delta
                return arm_joint_des, raw_delta, raw_delta

            def fallback_position_dls(command_b, ee_pos_b, jac, arm_joint_pos):
                jac_pos = jac[:, 0:3, :]
                delta_pos = (command_b[:, 0:3] - ee_pos_b).unsqueeze(-1)
                eye = torch.eye(3, device=robot.device, dtype=jac_pos.dtype).unsqueeze(0)
                damping = float(args.ik_dls_damping)
                lhs = jac_pos @ jac_pos.transpose(-1, -2) + (damping * damping) * eye
                rhs = delta_pos
                delta_task = torch.linalg.solve(lhs, rhs)
                delta_joint = jac_pos.transpose(-1, -2) @ delta_task
                return arm_joint_pos + delta_joint.squeeze(-1)

            def torch_quat_conj_wxyz(q):
                out = q.clone()
                out[:, 1:4] = -out[:, 1:4]
                return out

            def torch_quat_mul_wxyz(a, b):
                aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
                bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                return torch.stack((
                    aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                ), dim=-1)

            def orientation_error_axis_angle(command_quat_b, ee_quat_b):
                q_err = torch_quat_mul_wxyz(command_quat_b, torch_quat_conj_wxyz(ee_quat_b))
                q_err = q_err / torch.clamp(torch.linalg.norm(q_err, dim=-1, keepdim=True), min=1.0e-12)
                sign = torch.where(
                    q_err[:, 0:1] < 0.0,
                    -torch.ones_like(q_err[:, 0:1]),
                    torch.ones_like(q_err[:, 0:1]),
                )
                return 2.0 * sign * q_err[:, 1:4]

            def ik_orientation_weight_for_phase(phase):
                if phase == "LIFT" and float(args.ik_lift_orientation_weight) >= 0.0:
                    return max(0.0, float(args.ik_lift_orientation_weight))
                if phase == "HOLD" and float(args.ik_hold_orientation_weight) >= 0.0:
                    return max(0.0, float(args.ik_hold_orientation_weight))
                return max(0.0, float(args.ik_orientation_weight))

            def weighted_pose_dls(command_b, ee_pos_b, ee_quat_b, jac, arm_joint_pos, phase):
                pos_err = command_b[:, 0:3] - ee_pos_b
                ori_err = orientation_error_axis_angle(command_b[:, 3:7], ee_quat_b)
                pos_weight = max(0.0, float(args.ik_position_weight))
                z_weight = max(0.0, float(args.ik_z_position_weight))
                pos_weights = torch.tensor(
                    [pos_weight, pos_weight, z_weight],
                    device=robot.device,
                    dtype=jac.dtype,
                    ).reshape(1, 3)
                orient_weight = ik_orientation_weight_for_phase(phase)
                effective_orient_weight = orient_weight
                z_priority_error = max(0.0, float(args.ik_z_priority_error_m))
                z_err_abs = float(torch.max(torch.abs(pos_err[:, 2])).detach().cpu().item())
                if z_priority_error > 0.0 and z_err_abs > z_priority_error:
                    effective_orient_weight = min(
                        orient_weight,
                        max(0.0, float(args.ik_z_priority_orientation_weight)),
                    )
                if orient_weight <= 0.0:
                    arm_joint_des = fallback_position_dls(command_b, ee_pos_b, jac, arm_joint_pos)
                    return arm_joint_des, {
                        "position_error_b": pos_err.detach().cpu().tolist(),
                        "orientation_error_axis_angle_b": ori_err.detach().cpu().tolist(),
                        "position_weight": pos_weight,
                        "z_position_weight": z_weight,
                        "orientation_weight": orient_weight,
                        "effective_orientation_weight": effective_orient_weight,
                        "orientation_weight_phase": str(phase),
                        "z_priority_active": False,
                    }
                task_err = torch.cat((pos_weights * pos_err, effective_orient_weight * ori_err), dim=-1).unsqueeze(-1)
                jac_weighted = jac.clone()
                jac_weighted[:, 0:3, :] = jac_weighted[:, 0:3, :] * pos_weights.unsqueeze(-1)
                jac_weighted[:, 3:6, :] = jac_weighted[:, 3:6, :] * effective_orient_weight
                eye = torch.eye(6, device=robot.device, dtype=jac.dtype).unsqueeze(0)
                damping = float(args.ik_dls_damping)
                lhs = jac_weighted @ jac_weighted.transpose(-1, -2) + (damping * damping) * eye
                delta_task = torch.linalg.solve(lhs, task_err)
                delta_joint = jac_weighted.transpose(-1, -2) @ delta_task
                arm_joint_des = arm_joint_pos + delta_joint.squeeze(-1)
                return arm_joint_des, {
                    "position_error_b": pos_err.detach().cpu().tolist(),
                    "orientation_error_axis_angle_b": ori_err.detach().cpu().tolist(),
                    "position_weight": pos_weight,
                    "z_position_weight": z_weight,
                    "orientation_weight": orient_weight,
                    "effective_orientation_weight": effective_orient_weight,
                    "orientation_weight_phase": str(phase),
                    "z_priority_active": bool(effective_orient_weight < orient_weight),
                    "z_priority_error_m": z_priority_error,
                }

            def assert_finite_ik_state(label, **items):
                bad = {}
                for key, value in items.items():
                    if torch.is_tensor(value):
                        finite = bool(torch.isfinite(value).all().item())
                        sample = value.detach().cpu().reshape(-1)[:12].tolist()
                    else:
                        arr = np.asarray(value, dtype=np.float64)
                        finite = bool(np.isfinite(arr).all())
                        sample = arr.reshape(-1)[:12].tolist()
                    if not finite:
                        bad[key] = sample
                if bad:
                    raise RuntimeError(f"{label} non-finite IK state: {bad}")

            def joint_limit_diagnostics(arm_joint_pos):
                out = {
                    "joint_names": [joint_names[i] for i in arm_ids],
                    "position": arm_joint_pos[0].detach().cpu().tolist(),
                    "limit_source": None,
                    "lower": None,
                    "upper": None,
                    "min_margin_rad": None,
                    "near_limit": False,
                }
                limits = None
                for attr in ("soft_joint_pos_limits", "joint_pos_limits"):
                    candidate = getattr(robot.data, attr, None)
                    if candidate is not None:
                        limits = candidate
                        out["limit_source"] = attr
                        break
                if limits is None:
                    return out
                try:
                    lim = limits[:, arm_ids, :] if limits.ndim == 3 else limits[arm_ids, :]
                    if lim.ndim == 3:
                        lim = lim[0]
                    lower = lim[:, 0].detach().cpu()
                    upper = lim[:, 1].detach().cpu()
                    pos = arm_joint_pos[0].detach().cpu()
                    margin = torch.minimum(pos - lower, upper - pos)
                    out.update({
                        "lower": lower.tolist(),
                        "upper": upper.tolist(),
                        "min_margin_rad": float(margin.min().item()),
                        "near_limit": bool((margin < 0.05).any().item()),
                    })
                except Exception as limit_exc:
                    out["limit_error"] = f"{type(limit_exc).__name__}: {limit_exc}"
                return out

            def pose_error_summary(target_pos_w_np, target_quat_w):
                ee_pos_w, ee_quat_w = current_ee_pose_w()
                target_pos = np.asarray(target_pos_w_np, dtype=np.float64)
                actual_pos = ee_pos_w[0].detach().cpu().numpy().astype(np.float64)
                target_quat = np.asarray(target_quat_w[0].detach().cpu().tolist(), dtype=np.float64)
                actual_quat = np.asarray(ee_quat_w[0].detach().cpu().tolist(), dtype=np.float64)
                pos_err = target_pos - actual_pos
                return {
                    "pos_error_w": pos_err.tolist(),
                    "pos_error_norm_m": float(np.linalg.norm(pos_err)),
                    "z_error_m": float(pos_err[2]),
                    "orientation_error_rad": _quat_angle_error_wxyz(target_quat, actual_quat),
                }

            def solve_ik_joint_target(command_b, ee_pos_b, ee_quat_b, jac, arm_joint_pos, phase):
                solver = str(args.ik_solver)
                extra = {}
                if solver == "controller":
                    diff_ik.set_command(command_b)
                    try:
                        arm_joint_des = diff_ik.compute(
                            ee_pos_b,
                            ee_quat_b,
                            jac,
                            arm_joint_pos,
                        )
                    except Exception as ik_exc:
                        diag = jacobian_diagnostics(jac)
                        diag.update({
                            "source": "DifferentialIKController.compute",
                            "exception": f"{type(ik_exc).__name__}: {ik_exc}",
                            "command_b": command_b.detach().cpu().tolist(),
                            "ee_pos_b": ee_pos_b.detach().cpu().tolist(),
                            "arm_joint_pos": arm_joint_pos.detach().cpu().tolist(),
                        })
                        physics_probe.setdefault("ik_compute_failures", []).append(diag)
                        flog(
                            "IK_COMPUTE_FAIL "
                            f"{type(ik_exc).__name__}: {ik_exc} "
                            f"diag={diag}"
                        )
                        arm_joint_des = fallback_position_dls(command_b, ee_pos_b, jac, arm_joint_pos)
                        physics_probe["ik_controller_fallback"] = "position_only_damped_least_squares_after_controller_exception"
                        extra["fallback"] = physics_probe["ik_controller_fallback"]
                elif solver == "position-dls":
                    arm_joint_des = fallback_position_dls(command_b, ee_pos_b, jac, arm_joint_pos)
                elif solver == "weighted-dls":
                    arm_joint_des, extra = weighted_pose_dls(
                        command_b,
                        ee_pos_b,
                        ee_quat_b,
                        jac,
                        arm_joint_pos,
                        phase,
                    )
                else:
                    raise RuntimeError(f"unknown IK solver {solver!r}")
                arm_joint_des, raw_delta, applied_delta = clamp_joint_delta(arm_joint_pos, arm_joint_des)
                extra.update({
                    "solver": solver,
                    "raw_joint_delta": raw_delta[0].detach().cpu().tolist(),
                    "applied_joint_delta": applied_delta[0].detach().cpu().tolist(),
                    "max_abs_raw_joint_delta_rad": float(torch.max(torch.abs(raw_delta)).detach().cpu().item()),
                    "max_abs_applied_joint_delta_rad": float(torch.max(torch.abs(applied_delta)).detach().cpu().item()),
                    "joint_delta_clamped": bool(torch.any(torch.abs(raw_delta - applied_delta) > 1.0e-9).detach().cpu().item()),
                })
                return arm_joint_des, extra

            phase_gripper_write_failures = {}

            def phase_gripper_gains(phase):
                gains = {
                    "stiffness": float(effective_gripper_stiffness),
                    "damping": float(effective_gripper_damping),
                    "effort_limit": float(effective_gripper_effort_limit),
                }
                if phase == "LIFT":
                    if float(args.ik_lift_gripper_stiffness) >= 0.0:
                        gains["stiffness"] = float(args.ik_lift_gripper_stiffness)
                    if float(args.ik_lift_gripper_damping) >= 0.0:
                        gains["damping"] = float(args.ik_lift_gripper_damping)
                    if float(args.ik_lift_gripper_effort_limit) >= 0.0:
                        gains["effort_limit"] = float(args.ik_lift_gripper_effort_limit)
                elif phase == "HOLD":
                    if float(args.ik_hold_gripper_stiffness) >= 0.0:
                        gains["stiffness"] = float(args.ik_hold_gripper_stiffness)
                    if float(args.ik_hold_gripper_damping) >= 0.0:
                        gains["damping"] = float(args.ik_hold_gripper_damping)
                    if float(args.ik_hold_gripper_effort_limit) >= 0.0:
                        gains["effort_limit"] = float(args.ik_hold_gripper_effort_limit)
                return gains

            def apply_phase_gripper_gains(phase):
                gains = phase_gripper_gains(phase)
                diagnostics = {
                    "phase": str(phase),
                    "requested": dict(gains),
                    "applied": {},
                    "errors": {},
                }
                method_by_key = {
                    "stiffness": "write_joint_stiffness_to_sim",
                    "damping": "write_joint_damping_to_sim",
                    "effort_limit": "write_joint_effort_limit_to_sim",
                }
                for key, method_name in method_by_key.items():
                    method = getattr(robot, method_name, None)
                    if method is None:
                        diagnostics["applied"][key] = False
                        diagnostics["errors"][key] = f"missing {method_name}"
                        continue
                    failure_key = (method_name, tuple(gripper_ids))
                    if failure_key in phase_gripper_write_failures:
                        diagnostics["applied"][key] = False
                        diagnostics["errors"][key] = phase_gripper_write_failures[failure_key]
                        continue
                    value = torch.full((1, len(gripper_ids)), float(gains[key]), device=robot.device)
                    try:
                        method(value, joint_ids=gripper_ids)
                        diagnostics["applied"][key] = True
                    except Exception as gain_exc:
                        err = f"{type(gain_exc).__name__}: {gain_exc}"
                        phase_gripper_write_failures[failure_key] = err
                        diagnostics["applied"][key] = False
                        diagnostics["errors"][key] = err
                return diagnostics

            def write_ik_step(phase, target_pos_w_np, target_quat_w, gripper_target_value):
                last_diag = {}
                substeps = max(1, int(args.ik_control_substeps))
                for substep in range(substeps):
                    gripper_gain_diag = apply_phase_gripper_gains(phase)
                    command_b = ee_target_command_b(target_pos_w_np, target_quat_w)
                    ee_pos_b, ee_quat_b = current_ee_pose_b()
                    arm_joint_pos = robot.data.joint_pos[:, arm_ids]
                    jac = current_ee_jacobian()
                    assert_finite_ik_state(
                        "before_ik_compute",
                        command_b=command_b,
                        ee_pos_b=ee_pos_b,
                        ee_quat_b=ee_quat_b,
                        arm_joint_pos=arm_joint_pos,
                        jac=jac,
                    )
                    arm_joint_des, solve_diag = solve_ik_joint_target(
                        command_b,
                        ee_pos_b,
                        ee_quat_b,
                        jac,
                        arm_joint_pos,
                        phase,
                    )
                    assert_finite_ik_state("after_ik_compute", arm_joint_des=arm_joint_des)
                    robot.set_joint_position_target(arm_joint_des, joint_ids=arm_ids)
                    gripper_target_tensor = torch.full(
                        (1, len(gripper_ids)),
                        float(gripper_target_value),
                        device=robot.device,
                    )
                    robot.set_joint_position_target(gripper_target_tensor, joint_ids=gripper_ids)
                    robot.write_data_to_sim()
                    write_robot_root_hold()
                    write_pedestal_hold()
                    sim.step()
                    robot.update(1.0 / 120.0)
                    obj.update(1.0 / 120.0)
                    if pedestal is not None:
                        pedestal.update(1.0 / 120.0)
                    last_diag = {
                        "substep": int(substep + 1),
                        "substeps": int(substeps),
                        "command_b": command_b[0].detach().cpu().tolist(),
                        "jacobian": jacobian_diagnostics(jac),
                        "joint_limits": joint_limit_diagnostics(arm_joint_pos),
                        "arm_joint_target": arm_joint_des[0].detach().cpu().tolist(),
                        "gripper_phase_gains": gripper_gain_diag,
                        **solve_diag,
                    }
                last_diag["target_tracking"] = pose_error_summary(target_pos_w_np, target_quat_w)
                return last_diag

            # This renderer is deliberately separate from the post-hoc sequence writer:
            # the overlay follows the live physics rows but cannot alter NPZ/summary data.
            inline_overlay_timing = []
            inline_overlay_state = {}

            def update_inline_tactile_overlay(phase, step_in_phase, row):
                tactile_stride = max(1, int(args.tactile_every))
                overlay_stride = max(1, int(args.tactile_overlay_update_every))
                if (tactile_overlay is None or step_in_phase % tactile_stride
                        or (step_in_phase // tactile_stride) % overlay_stride):
                    return
                try:
                    if not inline_overlay_state:
                        inline_overlay_state["left"] = VBTSSensor(args.ckpt, device="cuda" if args.cuda else None)
                        inline_overlay_state["right"] = VBTSSensor(args.ckpt, device="cuda" if args.cuda else None)
                        inline_overlay_state["grid"] = gel_grid(inline_overlay_state["left"].side, 0.009)
                        inline_overlay_state["track_left"] = ShearTracker(mu=args.mu)
                        inline_overlay_state["track_right"] = ShearTracker(mu=args.mu)
                    object_pose = row["object_pose_w"]
                    T_w_obj = pose_to_T(object_pose["pos"], object_pose["quat_wxyz"])
                    images = {}
                    for side, tracker_key in (("left", "track_left"), ("right", "track_right")):
                        pad = next(p for p in manifest["pads"] if p["name"] == side)
                        pose = row["pad_pose_w"][side]
                        T_w_link = pose_to_T(pose["pos"], pose["quat_wxyz"])
                        T_gel_obj = T_inv(T_w_link @ np.asarray(pad["T_link_gel"], dtype=np.float64)) @ T_w_obj
                        pen = _pen_for_object(args.object, args.object_size, T_gel_obj, inline_overlay_state["grid"])
                        sx, sy = inline_overlay_state[tracker_key].update(T_gel_obj, pen)
                        images[side] = inline_overlay_state[side].render(
                            pen, float(sx), float(sy), mu=args.mu, E=args.E, noise=not args.no_noise
                        )["img"][0, 0].detach().cpu().numpy()
                    started = time.perf_counter()
                    tactile_overlay.update(images["left"], images["right"])
                    inline_overlay_timing.append(time.perf_counter() - started)
                except Exception as exc:
                    tactile_overlay.enabled = False
                    flog(f"TACTILE_OVERLAY_FAIL inline {type(exc).__name__}: {exc}")

            def append_ik_row(phase, target_pos_w_np, target_quat_w, gripper_target_value, step_in_phase, ik_diag=None):
                row = physics_trace_row(phase, gripper_target_value)
                ee_pos_w, ee_quat_w = current_ee_pose_w()
                row["step_in_phase"] = int(step_in_phase)
                row["grasp_mode"] = "ik"
                row["ee_body_name"] = body_names[ee_body_id]
                row["ee_pose_w"] = {
                    "pos": ee_pos_w[0].detach().cpu().tolist(),
                    "quat_wxyz": ee_quat_w[0].detach().cpu().tolist(),
                }
                row["ee_target_pose_w"] = {
                    "pos": np.asarray(target_pos_w_np, dtype=np.float64).tolist(),
                    "quat_wxyz": target_quat_w[0].detach().cpu().tolist(),
                }
                row["arm_joint_pos"] = robot.data.joint_pos[0, arm_ids].detach().cpu().tolist()
                row["ik_diagnostics"] = ik_diag or {}
                row["ik_error"] = pose_error_summary(target_pos_w_np, target_quat_w)
                physics_trace.append(row)
                update_inline_tactile_overlay(phase, int(step_in_phase), row)
                label = world_camera_capture_label(phase, int(step_in_phase))
                if label is not None:
                    capture_world_camera_frame(label, row)
                return row

            ik_close_floor_phases = {"DESCEND", "ALIGN", "CLOSE", "SETTLE"}

            def ik_close_min_ee_z(close_target_z=None):
                floors = []
                z = float(args.ik_min_close_ee_z)
                if np.isfinite(z) and z > 0.0:
                    floors.append(z)
                if close_target_z is not None:
                    close_z = float(close_target_z)
                    if np.isfinite(close_z):
                        floors.append(close_z)
                return max(floors) if floors else None

            def clamp_close_target_to_floor(phase, nominal_target_pos, target_pos, close_target_z=None):
                min_z = ik_close_min_ee_z(close_target_z)
                if min_z is None or phase not in ik_close_floor_phases:
                    return None
                if float(target_pos[2]) >= min_z:
                    return None
                unclamped_z = float(target_pos[2])
                target_pos[2] = min_z
                ik_live_target_bias_w[2] = float(target_pos[2] - nominal_target_pos[2])
                return {
                    "enabled": True,
                    "phase": str(phase),
                    "min_ee_z_m": float(min_z),
                    "unclamped_target_z_m": float(unclamped_z),
                    "clamped_target_z_m": float(target_pos[2]),
                    "target_bias_w": ik_live_target_bias_w.tolist(),
                }

            def run_ik_phase(
                phase,
                start_pos_w_np,
                end_pos_w_np,
                target_quat_w,
                steps,
                gripper_start_value,
                gripper_end_value=None,
                stop_on_dual_force=False,
                gel_midpoint_target_w=None,
                gel_midpoint_end_w=None,
                live_midpoint_servo=False,
            ):
                phase_rows = []
                steps = int(max(steps, 0))
                gripper_end = float(gripper_start_value if gripper_end_value is None else gripper_end_value)
                prev_finger = None
                stall_count = 0
                stopped_reason = "completed"
                servo_target_start = None if gel_midpoint_target_w is None else np.asarray(gel_midpoint_target_w, dtype=np.float64)
                servo_target_end = (
                    None
                    if gel_midpoint_end_w is None
                    else np.asarray(gel_midpoint_end_w, dtype=np.float64)
                )
                phase_close_target_z = (
                    float(np.asarray(end_pos_w_np, dtype=np.float64)[2])
                    if phase in ik_close_floor_phases
                    else None
                )
                for k in range(steps):
                    alpha = (k + 1) / float(max(steps, 1))
                    nominal_target_pos = (
                        (1.0 - alpha) * np.asarray(start_pos_w_np, dtype=np.float64)
                        + alpha * np.asarray(end_pos_w_np, dtype=np.float64)
                    )
                    target_pos = nominal_target_pos + ik_live_target_bias_w
                    floor_clamp_diag = clamp_close_target_to_floor(
                        phase, nominal_target_pos, target_pos, phase_close_target_z
                    )
                    servo_diag = None
                    if (
                        live_midpoint_servo
                        and servo_target_start is not None
                        and np.isfinite(servo_target_start).all()
                    ):
                        servo_target = servo_target_start
                        if servo_target_end is not None and np.isfinite(servo_target_end).all():
                            servo_target = (
                                (1.0 - alpha) * servo_target_start
                                + alpha * servo_target_end
                            )
                        measured_mid = face_midpoint_from_state(gel_gap_state())
                        if measured_mid is not None:
                            measured_mid = np.asarray(measured_mid, dtype=np.float64)
                            if np.isfinite(measured_mid).all():
                                midpoint_error = servo_target - measured_mid
                                correction = midpoint_error * float(args.ik_live_midpoint_servo_gain)
                                max_step = max(0.0, float(args.ik_live_midpoint_servo_max_step))
                                correction_norm = float(np.linalg.norm(correction))
                                if max_step > 0.0 and correction_norm > max_step:
                                    correction = correction * (max_step / (correction_norm + 1.0e-12))
                                ik_live_target_bias_w[:] = ik_live_target_bias_w + correction
                                max_total = max(0.0, float(args.ik_live_midpoint_servo_max_total))
                                total_norm = float(np.linalg.norm(ik_live_target_bias_w))
                                if max_total > 0.0 and total_norm > max_total:
                                    ik_live_target_bias_w[:] = ik_live_target_bias_w * (
                                        max_total / (total_norm + 1.0e-12)
                                    )
                                target_pos = nominal_target_pos + ik_live_target_bias_w
                                floor_clamp_diag = (
                                    clamp_close_target_to_floor(
                                        phase, nominal_target_pos, target_pos, phase_close_target_z
                                    )
                                    or floor_clamp_diag
                                )
                                servo_diag = {
                                    "enabled": True,
                                    "target_midpoint_w": servo_target.tolist(),
                                    "measured_midpoint_w": measured_mid.tolist(),
                                    "midpoint_error_w": midpoint_error.tolist(),
                                    "applied_correction_w": correction.tolist(),
                                    "target_bias_w": ik_live_target_bias_w.tolist(),
                                }
                    gripper_value = (
                        (1.0 - alpha) * float(gripper_start_value)
                        + alpha * gripper_end
                    )
                    if phase == "LIFT":
                        gripper_value += float(args.ik_lift_gripper_target_offset)
                    elif phase == "HOLD":
                        gripper_value += float(args.ik_hold_gripper_target_offset)
                    ik_diag = write_ik_step(phase, target_pos, target_quat_w, gripper_value)
                    if servo_diag is not None:
                        ik_diag["live_midpoint_servo"] = servo_diag
                    if floor_clamp_diag is not None:
                        ik_diag["workspace_floor_clamp"] = floor_clamp_diag
                    row = append_ik_row(phase, target_pos, target_quat_w, gripper_value, k + 1, ik_diag)
                    phase_rows.append(row)
                    log_every = int(args.ik_diagnostics_log_every)
                    if log_every > 0 and (
                        k == 0 or (k + 1) == steps or (k + 1) % log_every == 0
                    ):
                        err = row.get("ik_error", {})
                        jac_diag = (row.get("ik_diagnostics", {}) or {}).get("jacobian", {})
                        joint_diag = (row.get("ik_diagnostics", {}) or {}).get("joint_limits", {})
                        flog(
                            "ik_track "
                            f"phase={phase} step={k + 1}/{steps} "
                            f"target_z={float(target_pos[2]):.5f} "
                            f"z_err={float(err.get('z_error_m', float('nan'))):.5f} "
                            f"pos_err={float(err.get('pos_error_norm_m', float('nan'))):.5f} "
                            f"ori_err={float(err.get('orientation_error_rad', float('nan'))):.5f} "
                            f"jac_rank={jac_diag.get('rank')} jac_cond={jac_diag.get('condition')} "
                            f"near_joint_limit={joint_diag.get('near_limit')}"
                        )
                    joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                    finger_pos = float(joint_pos[gripper_ids[0]]) if gripper_ids else None
                    if prev_finger is not None and finger_pos is not None:
                        if abs(finger_pos - prev_finger) < args.physics_close_stall_eps:
                            stall_count += 1
                        else:
                            stall_count = 0
                    prev_finger = finger_pos
                    forces = row.get("force_N", {})
                    dual_force = all(
                        float(forces.get(side, 0.0)) >= float(args.ik_close_force_gate_N)
                        for side in ("left", "right")
                    )
                    if stop_on_dual_force and dual_force:
                        stopped_reason = "dual_force_gate"
                        row["stopped_reason"] = stopped_reason
                        break
                    if stop_on_dual_force and stall_count >= int(args.physics_close_stall_steps):
                        stopped_reason = "finger_joint_stall"
                        row["stopped_reason"] = stopped_reason
                        break
                return phase_rows, stopped_reason

            set_gripper(args.gripper_open_target, max(1, int(args.ik_home_steps)))
            home_ee_pos_w, home_ee_quat_w = current_ee_pose_w()
            home_pos = np.asarray(home_ee_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
            gel_state_open = gel_gap_state()
            ik_target_quat_w = home_ee_quat_w
            current_grip_axis = grip_axis_for_gel_frames(gel_state_open.get("frames", {}))
            desired_grip_axis = _vec3_arg(args.ik_grip_axis_world)
            desired_grip_axis = desired_grip_axis / (np.linalg.norm(desired_grip_axis) + 1.0e-12)
            if current_grip_axis is not None and np.isfinite(desired_grip_axis).all():
                home_quat_np = home_ee_quat_w[0].detach().cpu().numpy()
                q_delta = _quat_between_vectors_wxyz(current_grip_axis, desired_grip_axis)
                target_quat_np = _quat_mul_wxyz(q_delta, home_quat_np)
                ik_target_quat_w = torch.tensor(target_quat_np, device=robot.device, dtype=torch.float32).reshape(1, 4)
            def face_midpoint_from_state(state):
                frames = state.get("frames", {}) if isinstance(state, dict) else {}
                if "left" not in frames or "right" not in frames:
                    return None
                left = np.asarray(frames["left"].get("origin"), dtype=np.float64)
                right = np.asarray(frames["right"].get("origin"), dtype=np.float64)
                left_n = np.asarray(frames["left"].get("normal"), dtype=np.float64)
                right_n = np.asarray(frames["right"].get("normal"), dtype=np.float64)
                if not (
                    np.isfinite(left).all()
                    and np.isfinite(right).all()
                    and np.isfinite(left_n).all()
                    and np.isfinite(right_n).all()
                ):
                    return None
                left_n = left_n / (np.linalg.norm(left_n) + 1.0e-12)
                right_n = right_n / (np.linalg.norm(right_n) + 1.0e-12)
                axis = left_n - right_n
                if np.linalg.norm(axis) <= 1.0e-9:
                    axis = right - left
                axis = axis / (np.linalg.norm(axis) + 1.0e-12)
                origin_mid = 0.5 * (left + right)
                face_coord = 0.5 * (float(axis @ left) + float(axis @ right))
                return origin_mid + axis * (face_coord - float(axis @ origin_mid))

            def rotation_from_quat_wxyz(q_wxyz):
                return pose_to_T((0.0, 0.0, 0.0), np.asarray(q_wxyz, dtype=np.float64).tolist())[:3, :3]

            open_midpoint = face_midpoint_from_state(gel_state_open)
            if open_midpoint is None:
                open_midpoint = gel_state_open.get("midpoint")
            open_midpoint = None if open_midpoint is None else np.asarray(open_midpoint, dtype=np.float64)
            home_quat_np = np.asarray(home_ee_quat_w[0].detach().cpu().tolist(), dtype=np.float64)
            target_quat_np = np.asarray(ik_target_quat_w[0].detach().cpu().tolist(), dtype=np.float64)
            if open_midpoint is not None and np.isfinite(open_midpoint).all():
                gel_midpoint_offset_ee = rotation_from_quat_wxyz(home_quat_np).T @ (open_midpoint - home_pos)
                target_gel_midpoint_offset_w = rotation_from_quat_wxyz(target_quat_np) @ gel_midpoint_offset_ee
            else:
                gel_midpoint_offset_ee = None
                target_gel_midpoint_offset_w = None

            def ee_target_for_gel_midpoint(midpoint_w):
                midpoint_w = np.asarray(midpoint_w, dtype=np.float64)
                if target_gel_midpoint_offset_w is not None and np.isfinite(target_gel_midpoint_offset_w).all():
                    return midpoint_w - target_gel_midpoint_offset_w
                if open_midpoint is not None and np.isfinite(open_midpoint).all():
                    return home_pos + (midpoint_w - open_midpoint)
                return home_pos.copy()

            def target_gel_midpoint_for_ee(ee_pos_w):
                ee_pos_w = np.asarray(ee_pos_w, dtype=np.float64)
                if target_gel_midpoint_offset_w is not None and np.isfinite(target_gel_midpoint_offset_w).all():
                    return ee_pos_w + target_gel_midpoint_offset_w
                if open_midpoint is not None and np.isfinite(open_midpoint).all():
                    return open_midpoint + (ee_pos_w - home_pos)
                return ee_pos_w.copy()

            if open_midpoint is None:
                object_center = np.array([0.45, 0.0, scene_pedestal_height + args.object_size / 2.0], dtype=np.float64)
            else:
                object_center = np.asarray(open_midpoint, dtype=np.float64)
            object_center = object_center + _vec3_arg(args.ik_object_offset_world)
            pedestal_object_z = float(scene_pedestal_height + args.object_size / 2.0)
            if not np.isfinite(object_center).all():
                object_center = np.array([0.45, 0.0, pedestal_object_z], dtype=np.float64)
            object_center[2] = pedestal_object_z
            physics_probe["object_initial_center_w"] = object_center.tolist()
            physics_probe["object_initialization"] = {
                "mode": "pedestal_top_under_open_gel_midpoint_xy",
                "pedestal_top_z_m": float(scene_pedestal_height),
                "pedestal_height_source": scene_pedestal_height_source,
                "object_center_z_m": float(object_center[2]),
                "object_size_m": float(args.object_size),
            }
            physics_probe["open_gel_gap_state"] = gel_state_open
            staging_center = np.array([2.0, 2.0, pedestal_object_z], dtype=np.float64)
            physics_probe["pedestal_staging"] = place_pedestal_under(staging_center)
            place_object_world(staging_center.tolist(), quat=(1.0, 0.0, 0.0, 0.0))

            descend_pos = ee_target_for_gel_midpoint(object_center)
            initial_descend_pos = descend_pos.copy()
            descend_target_source = "static_open_midpoint_offset"
            min_close_z = ik_close_min_ee_z()
            if min_close_z is not None and float(descend_pos[2]) < min_close_z:
                descend_pos[2] = min_close_z
                descend_target_source = "static_open_midpoint_offset_clamped_to_reachable_floor"
            pregrasp_pos = descend_pos + np.array([0.0, 0.0, float(args.ik_pregrasp_height)], dtype=np.float64)
            lift_pos = descend_pos + np.array([0.0, 0.0, float(args.physics_lift_height)], dtype=np.float64)
            ik_live_target_bias_w = np.zeros(3, dtype=np.float64)
            physics_probe["ik"] = {
                "controller": "isaaclab.controllers.DifferentialIKController",
                "ik_method": "dls",
                "solver": str(args.ik_solver),
                "orientation_weight": float(args.ik_orientation_weight),
                "lift_orientation_weight": (
                    None if float(args.ik_lift_orientation_weight) < 0.0
                    else float(args.ik_lift_orientation_weight)
                ),
                "hold_orientation_weight": (
                    None if float(args.ik_hold_orientation_weight) < 0.0
                    else float(args.ik_hold_orientation_weight)
                ),
                "z_priority_orientation_weight": float(args.ik_z_priority_orientation_weight),
                "z_priority_error_m": float(args.ik_z_priority_error_m),
                "control_substeps": int(args.ik_control_substeps),
                "max_joint_step_rad": float(args.ik_max_joint_step),
                "ee_body_name": body_names[ee_body_id],
                "arm_joint_names": [joint_names[i] for i in arm_ids],
                "gripper_joint_names": [joint_names[i] for i in gripper_ids],
                "gripper_target_offsets": {
                    "lift": float(args.ik_lift_gripper_target_offset),
                    "hold": float(args.ik_hold_gripper_target_offset),
                },
                "gripper_phase_gains": {
                    "close": {
                        "stiffness": float(effective_gripper_stiffness),
                        "damping": float(effective_gripper_damping),
                        "effort_limit": float(effective_gripper_effort_limit),
                    },
                    "lift": phase_gripper_gains("LIFT"),
                    "hold": phase_gripper_gains("HOLD"),
                },
                "object_center_w": object_center.tolist(),
                "home_ee_pos_w": home_pos.tolist(),
                "pregrasp_ee_pos_w": pregrasp_pos.tolist(),
                "descend_ee_pos_w": descend_pos.tolist(),
                "initial_static_descend_ee_pos_w": initial_descend_pos.tolist(),
                "descend_target_source": descend_target_source,
                "min_close_ee_z_m": None if min_close_z is None else float(min_close_z),
                "lift_ee_pos_w": lift_pos.tolist(),
                "home_ee_quat_wxyz": home_ee_quat_w[0].detach().cpu().tolist(),
                "ee_quat_wxyz": ik_target_quat_w[0].detach().cpu().tolist(),
                "grip_axis_alignment": {
                    "source_axis_w": None if current_grip_axis is None else np.asarray(current_grip_axis, dtype=np.float64).tolist(),
                    "target_axis_w": desired_grip_axis.tolist(),
                },
                "gel_midpoint_targeting": {
                    "open_midpoint_w": None if open_midpoint is None else open_midpoint.tolist(),
                    "offset_ee": None if gel_midpoint_offset_ee is None else gel_midpoint_offset_ee.tolist(),
                    "target_offset_w": None if target_gel_midpoint_offset_w is None else target_gel_midpoint_offset_w.tolist(),
                    "descend_target_gel_midpoint_w": target_gel_midpoint_for_ee(descend_pos).tolist(),
                    "cube_center_z_m": float(pedestal_object_z),
                    "descend_target_vertical_gap_m": float(target_gel_midpoint_for_ee(descend_pos)[2] - pedestal_object_z),
                    "mode": "live_measured_midpoint_servo" if args.ik_live_midpoint_servo else "static_open_offset",
                    "live_servo_enabled": bool(args.ik_live_midpoint_servo),
                    "live_servo_close_enabled": bool(args.ik_live_midpoint_servo_close),
                    "lift_live_servo_enabled": bool(args.ik_lift_live_midpoint_servo),
                    "live_servo_gain": float(args.ik_live_midpoint_servo_gain),
                    "live_servo_max_step_m": float(args.ik_live_midpoint_servo_max_step),
                    "live_servo_max_total_m": float(args.ik_live_midpoint_servo_max_total),
                },
                "pregrasp_height_m": float(args.ik_pregrasp_height),
                "close_force_gate_N": float(args.ik_close_force_gate_N),
                "dynamic_object": True,
                "kinematic_object_hold": False,
            }

            diff_ik.reset()
            home_rows, _ = run_ik_phase(
                "HOME",
                home_pos,
                home_pos,
                ik_target_quat_w,
                args.ik_home_steps,
                args.gripper_open_target,
            )
            pre_rows, _ = run_ik_phase(
                "PRE_GRASP",
                home_pos,
                pregrasp_pos,
                ik_target_quat_w,
                args.ik_pregrasp_steps,
                args.gripper_open_target,
            )
            pregrasp_live_state = gel_gap_state()
            pregrasp_live_midpoint = face_midpoint_from_state(pregrasp_live_state)
            if pregrasp_live_midpoint is None:
                pregrasp_live_midpoint = pregrasp_live_state.get("midpoint")
            pregrasp_live_retarget = {
                "enabled": False,
                "reason": "pregrasp live midpoint unavailable",
            }
            if pregrasp_live_midpoint is not None:
                measured_mid = np.asarray(pregrasp_live_midpoint, dtype=np.float64)
                pre_ee_pos_w, _ = current_ee_pose_w()
                measured_ee = np.asarray(pre_ee_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
                if np.isfinite(measured_mid).all() and np.isfinite(measured_ee).all():
                    live_midpoint_offset_w = measured_mid - measured_ee
                    live_descend_pos = object_center - live_midpoint_offset_w
                    unclamped_live_descend_pos = live_descend_pos.copy()
                    min_close_z = ik_close_min_ee_z()
                    if min_close_z is not None and float(live_descend_pos[2]) < min_close_z:
                        live_descend_pos[2] = min_close_z
                    descend_pos = live_descend_pos
                    lift_pos = descend_pos + np.array([0.0, 0.0, float(args.physics_lift_height)], dtype=np.float64)
                    descend_target_source = "pregrasp_live_midpoint_offset"
                    if not np.allclose(unclamped_live_descend_pos, live_descend_pos):
                        descend_target_source = "pregrasp_live_midpoint_offset_clamped_to_reachable_floor"
                    pregrasp_live_retarget = {
                        "enabled": True,
                        "mode": "pregrasp_live_midpoint_offset_to_shelf_cube",
                        "pregrasp_measured_midpoint_w": measured_mid.tolist(),
                        "pregrasp_measured_ee_w": measured_ee.tolist(),
                        "live_midpoint_offset_w": live_midpoint_offset_w.tolist(),
                        "unclamped_descend_ee_pos_w": unclamped_live_descend_pos.tolist(),
                        "descend_ee_pos_w": descend_pos.tolist(),
                        "min_close_ee_z_m": None if min_close_z is None else float(min_close_z),
                    }
                    physics_probe["ik"]["descend_ee_pos_w"] = descend_pos.tolist()
                    physics_probe["ik"]["lift_ee_pos_w"] = lift_pos.tolist()
                    physics_probe["ik"]["descend_target_source"] = descend_target_source
                    physics_probe["ik"]["gel_midpoint_targeting"]["descend_target_gel_midpoint_w"] = (
                        target_gel_midpoint_for_ee(descend_pos).tolist()
                    )
                    physics_probe["ik"]["gel_midpoint_targeting"]["descend_target_vertical_gap_m"] = (
                        float(target_gel_midpoint_for_ee(descend_pos)[2] - pedestal_object_z)
                    )
                else:
                    pregrasp_live_retarget = {
                        "enabled": False,
                        "reason": "pregrasp live midpoint or ee pose non-finite",
                    }
            physics_probe["pregrasp_live_retarget"] = pregrasp_live_retarget
            physics_probe["pedestal"] = place_pedestal_under(object_center)
            place_object_world(object_center.tolist(), quat=(1.0, 0.0, 0.0, 0.0))
            for _ in range(max(0, int(args.ik_object_settle_steps))):
                robot.set_joint_position_target(
                    torch.full((1, len(gripper_ids)), float(args.gripper_open_target), device=robot.device),
                    joint_ids=gripper_ids,
                )
                robot.write_data_to_sim()
                write_robot_root_hold()
                write_pedestal_hold()
                sim.step()
                robot.update(1.0 / 120.0)
                obj.update(1.0 / 120.0)
                if pedestal is not None:
                    pedestal.update(1.0 / 120.0)
            descend_rows, _ = run_ik_phase(
                "DESCEND",
                pregrasp_pos,
                descend_pos,
                ik_target_quat_w,
                args.ik_descend_steps,
                args.gripper_open_target,
                gel_midpoint_target_w=object_center,
                live_midpoint_servo=bool(args.ik_live_midpoint_servo),
            )
            preclose_gel_state = gel_gap_state()
            align_rows = []
            preclose_midpoint_for_align = face_midpoint_from_state(preclose_gel_state)
            if preclose_midpoint_for_align is None:
                preclose_midpoint_for_align = preclose_gel_state.get("midpoint")
            if args.ik_live_midpoint_servo and preclose_midpoint_for_align is not None:
                measured_mid = np.asarray(preclose_midpoint_for_align, dtype=np.float64)
                measured_axis = grip_axis_for_gel_frames(preclose_gel_state.get("frames", {}))
                if measured_axis is not None and np.isfinite(measured_mid).all():
                    measured_axis = measured_axis / (np.linalg.norm(measured_axis) + 1.0e-12)
                    midpoint_error = object_center - measured_mid
                    align_rows, _ = run_ik_phase(
                        "ALIGN",
                        descend_pos,
                        descend_pos,
                        ik_target_quat_w,
                        args.ik_preclose_align_steps,
                        args.gripper_open_target,
                        gel_midpoint_target_w=object_center,
                        live_midpoint_servo=True,
                    )
                    post_align_mid = face_midpoint_from_state(gel_gap_state())
                    post_align_mid = None if post_align_mid is None else np.asarray(post_align_mid, dtype=np.float64)
                    physics_probe["preclose_midpoint_align"] = {
                        "enabled": True,
                        "mode": "iterative_live_measured_midpoint_servo",
                        "midpoint_before_w": measured_mid.tolist(),
                        "midpoint_after_w": None if post_align_mid is None else post_align_mid.tolist(),
                        "object_center_w": object_center.tolist(),
                        "grip_axis_w": measured_axis.tolist(),
                        "initial_error_w": midpoint_error.tolist(),
                        "final_error_w": (
                            None if post_align_mid is None else (object_center - post_align_mid).tolist()
                        ),
                        "final_target_bias_w": ik_live_target_bias_w.tolist(),
                        "steps": int(args.ik_preclose_align_steps),
                    }
                    physics_probe["preclose_tangent_align"] = physics_probe["preclose_midpoint_align"]
                    physics_probe["ik"]["gel_midpoint_targeting"]["align_overcommand_target_gel_midpoint_w"] = (
                        (target_gel_midpoint_for_ee(descend_pos + ik_live_target_bias_w)).tolist()
                    )
            elif preclose_midpoint_for_align is not None:
                measured_mid = np.asarray(preclose_midpoint_for_align, dtype=np.float64)
                measured_axis = grip_axis_for_gel_frames(preclose_gel_state.get("frames", {}))
                if measured_axis is not None and np.isfinite(measured_mid).all():
                    measured_axis = measured_axis / (np.linalg.norm(measured_axis) + 1.0e-12)
                    midpoint_error = object_center - measured_mid
                    align_delta = midpoint_error.copy()
                    align_norm = float(np.linalg.norm(align_delta))
                    max_align = float(args.ik_preclose_align_max_m)
                    if align_norm > max_align > 0.0:
                        align_delta = align_delta * (max_align / align_norm)
                    if np.linalg.norm(align_delta) > 1.0e-4:
                        aligned_descend_pos = descend_pos + align_delta
                        align_rows, _ = run_ik_phase(
                            "ALIGN",
                            descend_pos,
                            aligned_descend_pos,
                            ik_target_quat_w,
                            args.ik_preclose_align_steps,
                            args.gripper_open_target,
                        )
                        physics_probe["preclose_midpoint_align"] = {
                            "enabled": True,
                            "mode": "full_3d_midpoint_error",
                            "midpoint_before_w": measured_mid.tolist(),
                            "object_center_w": object_center.tolist(),
                            "grip_axis_w": measured_axis.tolist(),
                            "requested_delta_w": midpoint_error.tolist(),
                            "applied_delta_w": align_delta.tolist(),
                            "applied_vertical_delta_m": float(align_delta[2]),
                            "steps": int(args.ik_preclose_align_steps),
                        }
                        physics_probe["preclose_tangent_align"] = physics_probe["preclose_midpoint_align"]
                        descend_pos = aligned_descend_pos
                        lift_pos = descend_pos + np.array([0.0, 0.0, float(args.physics_lift_height)], dtype=np.float64)
                        physics_probe["ik"]["descend_ee_pos_w"] = descend_pos.tolist()
                        physics_probe["ik"]["lift_ee_pos_w"] = lift_pos.tolist()
                        physics_probe["ik"]["gel_midpoint_targeting"]["align_overcommand_target_gel_midpoint_w"] = (
                            target_gel_midpoint_for_ee(descend_pos).tolist()
                        )
                    else:
                        physics_probe["preclose_midpoint_align"] = {
                            "enabled": False,
                            "reason": "full 3D midpoint error below threshold",
                            "midpoint_before_w": measured_mid.tolist(),
                            "object_center_w": object_center.tolist(),
                            "grip_axis_w": measured_axis.tolist(),
                        }
                        physics_probe["preclose_tangent_align"] = physics_probe["preclose_midpoint_align"]
            if not align_rows and "preclose_midpoint_align" not in physics_probe:
                physics_probe["preclose_midpoint_align"] = {
                    "enabled": False,
                    "reason": "descend midpoint or grip axis unavailable",
                }
                physics_probe["preclose_tangent_align"] = physics_probe["preclose_midpoint_align"]
            preclose_gel_state = gel_gap_state()
            preclose_midpoint = face_midpoint_from_state(preclose_gel_state)
            if preclose_midpoint is None:
                preclose_midpoint = preclose_gel_state.get("midpoint")
            close_probe_rows = []
            reopen_rows = []
            close_probe_state = None
            close_probe_midpoint = None
            selected_close_probe_row = None
            actual_close_target = float(args.gripper_target)
            close_start_target = float(args.gripper_open_target)
            close_probe_enabled = bool(args.ik_close_midpoint_probe)
            close_steps_effective = int(max(args.physics_close_steps, args.ik_soft_close_steps))

            def select_close_target_from_probe(rows, contact_depth):
                candidates = []
                for row in rows:
                    state = row.get("gel_gap", {})
                    frames = state.get("frames", {}) if isinstance(state, dict) else {}
                    gap = state.get("gap_m") if isinstance(state, dict) else None
                    target = row.get("target")
                    if gap is None or target is None:
                        continue
                    gap = float(gap)
                    target = float(target)
                    if not (np.isfinite(gap) and np.isfinite(target)) or gap <= 0.0:
                        continue
                    required_gap, supports = support_required_gap_for_gel_frames(frames, contact_depth)
                    if required_gap is None or not np.isfinite(float(required_gap)):
                        required_gap = nominal_gap_for_contact_depth(contact_depth)
                    normal_dot = state.get("normal_dot")
                    normal_penalty = 0.0
                    if normal_dot is not None and np.isfinite(float(normal_dot)):
                        normal_penalty = max(0.0, float(normal_dot) + 0.95) * 0.02
                    score = abs(gap - float(required_gap)) + normal_penalty
                    candidates.append({
                        "row": row,
                        "target": target,
                        "gap_m": gap,
                        "required_gap_m": float(required_gap),
                        "gap_error_m": float(gap - float(required_gap)),
                        "supports": supports,
                        "normal_dot": None if normal_dot is None else float(normal_dot),
                        "score": float(score),
                    })
                if not candidates:
                    return None
                return min(candidates, key=lambda item: item["score"])

            if close_probe_enabled:
                staging_center = np.array(
                    [2.0, 2.0, max(float(args.object_size / 2.0), float(pedestal_object_z))],
                    dtype=np.float64,
                )
                physics_probe["preclose_close_midpoint_probe"] = {
                    "enabled": True,
                    "reason": "measure live gel midpoint at the closed finger target before placing the cube",
                    "object_staged_center_w": staging_center.tolist(),
                    "probe_steps": int(args.ik_close_midpoint_probe_steps),
                    "reopen_steps": int(args.ik_close_midpoint_reopen_steps),
                    "probe_gripper_target": float(args.gripper_target),
                }
                physics_probe["pedestal_staging_for_close_midpoint_probe"] = place_pedestal_under(staging_center)
                place_object_world(staging_center.tolist(), quat=(1.0, 0.0, 0.0, 0.0))
                close_probe_rows, _ = run_ik_phase(
                    "CLOSE_PROBE",
                    descend_pos,
                    descend_pos,
                    ik_target_quat_w,
                    args.ik_close_midpoint_probe_steps,
                    args.gripper_open_target,
                    args.gripper_target,
                    stop_on_dual_force=False,
                    gel_midpoint_target_w=None,
                    live_midpoint_servo=False,
                )
                close_probe_state = gel_gap_state()
                close_probe_midpoint = face_midpoint_from_state(close_probe_state)
                if close_probe_midpoint is None:
                    close_probe_midpoint = close_probe_state.get("midpoint")
                close_probe_midpoint = (
                    None
                    if close_probe_midpoint is None
                    else np.asarray(close_probe_midpoint, dtype=np.float64)
                )
                target_selection = None
                if bool(args.ik_close_target_from_probe):
                    target_selection = select_close_target_from_probe(
                        close_probe_rows,
                        float(args.ik_close_target_pen),
                    )
                    if target_selection is not None:
                        selected_close_probe_row = target_selection["row"]
                        selected_state = selected_close_probe_row.get("gel_gap", {})
                        selected_midpoint = face_midpoint_from_state(selected_state)
                        if selected_midpoint is None and isinstance(selected_state, dict):
                            selected_midpoint = selected_state.get("midpoint")
                        selected_midpoint = (
                            None
                            if selected_midpoint is None
                            else np.asarray(selected_midpoint, dtype=np.float64)
                        )
                        if selected_midpoint is not None and np.isfinite(selected_midpoint).all():
                            close_probe_state = selected_state
                            close_probe_midpoint = selected_midpoint
                            actual_close_target = float(target_selection["target"])
                physics_probe["preclose_close_midpoint_probe"].update({
                    "steps_run": int(len(close_probe_rows)),
                    "measured_closed_gel_midpoint_w": (
                        None if close_probe_midpoint is None else close_probe_midpoint.tolist()
                    ),
                    "gel_gap_state": close_probe_state,
                    "target_selection_enabled": bool(args.ik_close_target_from_probe),
                    "target_contact_depth_m": float(args.ik_close_target_pen),
                    "target_selection": target_selection,
                    "selected_gripper_target": float(actual_close_target),
                })
                reopen_target = float(args.gripper_open_target)
                if (
                    bool(args.ik_place_at_selected_close_target)
                    and selected_close_probe_row is not None
                ):
                    reopen_target = float(actual_close_target)
                    close_start_target = float(actual_close_target)
                    physics_probe["preclose_close_midpoint_probe"]["placement_gripper_target"] = reopen_target
                reopen_rows, _ = run_ik_phase(
                    "REOPEN",
                    descend_pos,
                    descend_pos,
                    ik_target_quat_w,
                    args.ik_close_midpoint_reopen_steps,
                    args.gripper_target,
                    reopen_target,
                    stop_on_dual_force=False,
                    gel_midpoint_target_w=None,
                    live_midpoint_servo=False,
                )
                physics_probe["preclose_close_midpoint_probe"]["reopen_steps_run"] = int(len(reopen_rows))
                if (
                    bool(args.ik_place_at_selected_close_target)
                    and selected_close_probe_row is not None
                ):
                    reopened_state = gel_gap_state()
                    reopened_midpoint = face_midpoint_from_state(reopened_state)
                    if reopened_midpoint is None:
                        reopened_midpoint = reopened_state.get("midpoint")
                    reopened_midpoint = (
                        None
                        if reopened_midpoint is None
                        else np.asarray(reopened_midpoint, dtype=np.float64)
                    )
                    physics_probe["preclose_close_midpoint_probe"].update({
                        "post_reopen_gel_gap_state": reopened_state,
                        "post_reopen_gel_midpoint_w": (
                            None if reopened_midpoint is None else reopened_midpoint.tolist()
                        ),
                    })
                    if reopened_midpoint is not None and np.isfinite(reopened_midpoint).all():
                        close_probe_state = reopened_state
                        close_probe_midpoint = reopened_midpoint
            else:
                physics_probe["preclose_close_midpoint_probe"] = {
                    "enabled": False,
                    "reason": "disabled by --no-ik-close-midpoint-probe",
                }
            placement_midpoint = close_probe_midpoint
            placement_state = close_probe_state
            placement_source = (
                "post_reopen_selected_close_target_live_gel_midpoint"
                if bool(args.ik_place_at_selected_close_target) and selected_close_probe_row is not None
                else "closed_finger_live_gel_midpoint_probe"
            )
            if placement_midpoint is None or not np.isfinite(placement_midpoint).all():
                placement_midpoint = None if preclose_midpoint is None else np.asarray(preclose_midpoint, dtype=np.float64)
                placement_state = preclose_gel_state
                placement_source = "open_preclose_live_gel_midpoint_fallback"
            preclose_midpoint = placement_midpoint
            preclose_gel_state = placement_state
            if preclose_midpoint is not None:
                preclose_object_center = np.asarray(preclose_midpoint, dtype=np.float64)
                if np.isfinite(preclose_object_center).all():
                    preclose_object_center = preclose_object_center + _vec3_arg(args.ik_object_offset_world)
                    # V15 ride-up compensation: lift the cube z above the measured close-position
                    # gel midpoint so four-bar ride-up brings the pads down onto the cube center
                    # instead of over the top edge (which wedges the cube out). default 0 = legacy.
                    ride_up_offset = float(args.ik_place_ride_up_offset)
                    if ride_up_offset != 0.0:
                        preclose_object_center = preclose_object_center.copy()
                        preclose_object_center[2] = float(preclose_object_center[2]) + ride_up_offset
                    previous_pedestal_object_z = float(pedestal_object_z)
                    pedestal_object_z = float(preclose_object_center[2])
                    physics_probe["preclose_live_recenter"] = {
                        "enabled": True,
                        "reason": "align object/pedestal to measured gel midpoint at the close target before dynamic close",
                        "placement_source": placement_source,
                        "previous_object_center_w": object_center.tolist(),
                        "object_center_w": preclose_object_center.tolist(),
                        "object_offset_world": _vec3_arg(args.ik_object_offset_world).tolist(),
                        "previous_cube_center_z_m": previous_pedestal_object_z,
                        "measured_reachable_cube_center_z_m": float(preclose_object_center[2]),
                        "measured_reachable_pedestal_top_z_m": float(preclose_object_center[2] - args.object_size / 2.0),
                        "measured_closed_gel_midpoint_z_m": float(preclose_object_center[2] - ride_up_offset),
                        "ride_up_offset_m": ride_up_offset,
                        "vertical_target_gap_m": ride_up_offset,
                    "previous_target_bias_w": ik_live_target_bias_w.tolist(),
                    "target_bias_reset_for_close": True,
                    "selected_gripper_target": float(actual_close_target),
                    "gel_gap_state": preclose_gel_state,
                }
                    object_center = preclose_object_center
                    physics_probe["pedestal"] = place_pedestal_under(object_center)
                    place_object_world(object_center.tolist(), quat=(1.0, 0.0, 0.0, 0.0))
                    ik_live_target_bias_w[:] = 0.0
                    physics_probe["ik"]["object_center_w"] = object_center.tolist()
                    physics_probe["ik"]["gel_midpoint_targeting"]["cube_center_z_m"] = float(object_center[2])
                    physics_probe["ik"]["gel_midpoint_targeting"]["shelf_top_z_m"] = float(
                        object_center[2] - args.object_size / 2.0
                    )
                    physics_probe["ik"]["gel_midpoint_targeting"]["preclose_measured_shelf_mode"] = (
                        "object_z_set_to_live_gel_midpoint"
                    )
            else:
                physics_probe["preclose_live_recenter"] = {
                    "enabled": False,
                    "reason": "descend gel midpoint unavailable",
                }
            close_rows, close_reason = run_ik_phase(
                "CLOSE",
                descend_pos,
                descend_pos,
                ik_target_quat_w,
                close_steps_effective,
                close_start_target,
                actual_close_target,
                stop_on_dual_force=True,
                gel_midpoint_target_w=object_center,
                live_midpoint_servo=bool(args.ik_live_midpoint_servo and args.ik_live_midpoint_servo_close),
            )
            close_final_target = (
                float(close_rows[-1]["target"])
                if close_rows else float(actual_close_target)
            )
            settle_rows, _ = run_ik_phase(
                "SETTLE",
                descend_pos,
                descend_pos,
                ik_target_quat_w,
                args.physics_hold_steps,
                close_final_target,
                gel_midpoint_target_w=object_center,
                live_midpoint_servo=bool(args.ik_live_midpoint_servo and args.ik_live_midpoint_servo_close),
            )
            before_lift = obj.data.root_pos_w[0].detach().cpu().tolist()
            lift_midpoint_start = face_midpoint_from_state(gel_gap_state())
            if lift_midpoint_start is None:
                lift_midpoint_start = gel_gap_state().get("midpoint")
            lift_midpoint_start = (
                None
                if lift_midpoint_start is None
                else np.asarray(lift_midpoint_start, dtype=np.float64)
            )
            if lift_midpoint_start is None or not np.isfinite(lift_midpoint_start).all():
                lift_midpoint_start = object_center.copy()
            lift_midpoint_target = lift_midpoint_start + np.array(
                [0.0, 0.0, float(args.physics_lift_height)],
                dtype=np.float64,
            )
            physics_probe["ik"]["gel_midpoint_targeting"]["lift_start_midpoint_w"] = (
                lift_midpoint_start.tolist()
            )
            physics_probe["ik"]["gel_midpoint_targeting"]["lift_end_midpoint_target_w"] = (
                lift_midpoint_target.tolist()
            )
            lift_rows, _ = run_ik_phase(
                "LIFT",
                descend_pos,
                lift_pos,
                ik_target_quat_w,
                args.physics_lift_steps,
                close_final_target,
                gel_midpoint_target_w=lift_midpoint_start,
                gel_midpoint_end_w=lift_midpoint_target,
                live_midpoint_servo=bool(args.ik_lift_live_midpoint_servo),
            )
            hold_rows, _ = run_ik_phase(
                "HOLD",
                lift_pos,
                lift_pos,
                ik_target_quat_w,
                args.physics_hold_steps,
                close_final_target,
                gel_midpoint_target_w=lift_midpoint_target,
                live_midpoint_servo=bool(args.ik_lift_live_midpoint_servo),
            )
            after_lift = obj.data.root_pos_w[0].detach().cpu().tolist()

            def summarize_ik_tracking(rows):
                if not rows:
                    return {
                        "rows": 0,
                        "max_abs_z_error_m": None,
                        "final_z_error_m": None,
                        "max_pos_error_norm_m": None,
                        "max_orientation_error_rad": None,
                        "min_jacobian_rank": None,
                        "max_jacobian_condition": None,
                        "joint_delta_clamped_rows": 0,
                        "near_joint_limit_rows": 0,
                    }
                z_errors = [abs(float(r.get("ik_error", {}).get("z_error_m", 0.0))) for r in rows]
                pos_errors = [float(r.get("ik_error", {}).get("pos_error_norm_m", 0.0)) for r in rows]
                ori_errors = [float(r.get("ik_error", {}).get("orientation_error_rad", 0.0)) for r in rows]
                ranks = [
                    int(r.get("ik_diagnostics", {}).get("jacobian", {}).get("rank", -1))
                    for r in rows
                    if int(r.get("ik_diagnostics", {}).get("jacobian", {}).get("rank", -1)) >= 0
                ]
                conds = [
                    float(r.get("ik_diagnostics", {}).get("jacobian", {}).get("condition"))
                    for r in rows
                    if r.get("ik_diagnostics", {}).get("jacobian", {}).get("condition") is not None
                ]
                return {
                    "rows": int(len(rows)),
                    "max_abs_z_error_m": float(max(z_errors)) if z_errors else None,
                    "final_z_error_m": float(rows[-1].get("ik_error", {}).get("z_error_m", 0.0)),
                    "max_pos_error_norm_m": float(max(pos_errors)) if pos_errors else None,
                    "max_orientation_error_rad": float(max(ori_errors)) if ori_errors else None,
                    "min_jacobian_rank": int(min(ranks)) if ranks else None,
                    "max_jacobian_condition": float(max(conds)) if conds else None,
                    "joint_delta_clamped_rows": int(sum(
                        1 for r in rows if r.get("ik_diagnostics", {}).get("joint_delta_clamped")
                    )),
                    "near_joint_limit_rows": int(sum(
                        1 for r in rows
                        if r.get("ik_diagnostics", {}).get("joint_limits", {}).get("near_limit")
                    )),
                }

            def count_dual_contact(rows, force_gate, pen_gate=None):
                count = 0
                for row in rows:
                    forces = row.get("force_N", {})
                    pens = row.get("pen_stat_m") or row.get("pen_m", {})
                    force_ok = all(
                        float(forces.get(side, 0.0)) >= float(force_gate)
                        for side in ("left", "right")
                    )
                    pen_ok = True if pen_gate is None else all(
                        float(pens.get(side, 0.0)) > float(pen_gate)
                        for side in ("left", "right")
                    )
                    if force_ok and pen_ok:
                        count += 1
                return int(count)

            recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
            settle_dual_contact_rows = count_dual_contact(settle_rows, args.ik_close_force_gate_N)
            lift_pen_rows = count_dual_contact(lift_rows + hold_rows, 0.0, recess)
            settle_summary = summarize_phase_rows(settle_rows)
            lift_summary = summarize_phase_rows(lift_rows)
            hold_summary = summarize_phase_rows(hold_rows)
            object_lift_m = float(after_lift[2] - before_lift[2])
            max_force = max(
                float(settle_summary.get("max_force_L_N", 0.0)),
                float(settle_summary.get("max_force_R_N", 0.0)),
                float(lift_summary.get("max_force_L_N", 0.0)),
                float(lift_summary.get("max_force_R_N", 0.0)),
                float(hold_summary.get("max_force_L_N", 0.0)),
                float(hold_summary.get("max_force_R_N", 0.0)),
            )
            physics_probe.update({
                "home": {"steps_run": len(home_rows)},
                "pre_grasp": {
                    "steps_run": len(pre_rows),
                    "ik_tracking": summarize_ik_tracking(pre_rows),
                },
                "descend": {
                    "steps_run": len(descend_rows),
                    "ik_tracking": summarize_ik_tracking(descend_rows),
                },
                "align": {
                    "steps_run": len(align_rows),
                    "ik_tracking": summarize_ik_tracking(align_rows),
                },
                "close_midpoint_probe": {
                    "steps_run": len(close_probe_rows),
                    "reopen_steps_run": len(reopen_rows),
                    "ik_tracking": summarize_ik_tracking(close_probe_rows),
                },
                "close": {
                    "steps_run": len(close_rows),
                    "stopped_reason": close_reason,
                    "final_target": float(close_final_target),
                    "summary": summarize_phase_rows(close_rows),
                    "ik_tracking": summarize_ik_tracking(close_rows),
                },
                "settle": {
                    "steps_run": len(settle_rows),
                    "dual_contact_rows": int(settle_dual_contact_rows),
                    "dual_contact_force_gate_N": float(args.ik_close_force_gate_N),
                    "summary": settle_summary,
                    "ik_tracking": summarize_ik_tracking(settle_rows),
                },
                "lift": {
                    "steps_run": len(lift_rows),
                    "height_command_m": float(args.physics_lift_height),
                    "object_lift_m": object_lift_m,
                    "dual_pen_rows_lift_plus_hold": int(lift_pen_rows),
                    "summary": lift_summary,
                    "ik_tracking": summarize_ik_tracking(lift_rows),
                },
                "hold": {
                    "steps_run": len(hold_rows),
                    "summary": hold_summary,
                    "ik_tracking": summarize_ik_tracking(hold_rows),
                },
                "object_pos_before_lift": before_lift,
                "object_pos_after_lift": after_lift,
                "object_lift_m": object_lift_m,
                "contact_force": contact_force_info,
                "passive_joint_names": passive_log_names,
                "trace": physics_trace,
                "force_spike_limit_N": float(args.physics_force_spike_limit_N),
                "max_force_N": float(max_force),
                "force_spike_pass": bool(max_force < args.physics_force_spike_limit_N),
            })
            physics_probe["round3_gate_pass"] = bool(
                settle_dual_contact_rows > 0
                and len(lift_rows) >= int(args.physics_lift_steps)
                and object_lift_m >= 0.08
                and lift_pen_rows >= 100
                and physics_probe["force_spike_pass"]
            )
            physics_probe["p4b_gate_pass"] = bool(physics_probe["round3_gate_pass"])
        elif args.physics_lift_probe and args.object_at_pads and gripper_ids:
            physics_probe = {"enabled": True}
            if args.physics_place_closed_probe:
                set_gripper(args.gripper_target, 30)
                physics_contact_center_np = object_center_from_live_pads()
                physics_place_contact_depth = args.live_contact_depth
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
                    close_start_target, refined_live_state = refine_calibrated_close_target(
                        calib, close_start_target, live_place_depth
                    )
                    raw_contact_center_np = object_center_from_live_pads(contact_depth=live_place_depth)
                    physics_place_contact_depth = live_place_depth
                    live_state = refined_live_state if refined_live_state is not None else gel_gap_state()
                    physics_contact_center_np, live_state, center_balance = balance_live_contact_center(
                        raw_contact_center_np,
                        live_place_depth,
                        live_state,
                    )
                    live_pen_stats = predicted_contact_stats_for_gel_frames(
                        physics_contact_center_np, live_state["frames"]
                    )
                    live_pred = pen_stat_values(live_pen_stats)
                    live_pen_max = pen_max_values(live_pen_stats)
                    live_vals = [float(live_pred[side]) for side in ("left", "right") if side in live_pred]
                    live_stats = placement_pen_summary(live_pred, live_place_depth)
                    place_row["place_center"] = physics_contact_center_np.tolist()
                    place_row["raw_place_center"] = raw_contact_center_np.tolist()
                    place_row["center_balance"] = center_balance
                    place_row["predicted_pen_m"] = live_pred
                    place_row["predicted_pen_stat_m"] = live_pred
                    place_row["predicted_pen_max_m"] = live_pen_max
                    place_row["predicted_pen_stats"] = live_pen_stats
                    place_row["predicted_min_pen_m"] = float(min(live_vals)) if live_vals else 0.0
                    place_row["predicted_max_pen_m"] = float(max(live_vals)) if live_vals else 0.0
                    place_row["predicted_mean_pen_m"] = live_stats["mean_pen_m"]
                    place_row["predicted_imbalance_m"] = live_stats["imbalance_m"]
                    place_row["live_gap_m"] = live_state["gap_m"]
                    place_row["live_midpoint"] = live_state["midpoint"]
                    place_row["live_frames"] = live_state["frames"]
                    place_row["refined_target"] = float(close_start_target)
                    calib["place_center"] = physics_contact_center_np.tolist()
                    calib["raw_place_center"] = raw_contact_center_np.tolist()
                    calib["live_center_balance"] = center_balance
                    calib["live_place_depth_m"] = live_place_depth
                    calib["live_refined_target"] = float(close_start_target)
                else:
                    physics_contact_center_np = object_center_from_live_pads(contact_depth=-args.physics_open_clearance)
                    physics_place_contact_depth = -args.physics_open_clearance
                    close_start_target = args.gripper_open_target
            physics_probe["initial_planned_contact_center"] = physics_contact_center_np.tolist()
            physics_probe["planned_contact_center"] = physics_contact_center_np.tolist()
            physics_probe["initial_planned_contact_center_diagnostics"] = placement_center_diagnostics(
                physics_contact_center_np,
                physics_place_contact_depth,
            )
            physics_probe["place_closed_probe"] = bool(args.physics_place_closed_probe)
            physics_probe["pedestal_probe"] = bool(args.physics_pedestal_probe)
            physics_probe["open_clearance_m"] = 0.0 if args.physics_place_closed_probe else float(args.physics_open_clearance)
            physics_probe["retention_control"] = {
                "object_max_depenetration_velocity": (
                    None if effective_object_max_depenetration_velocity < 0.0
                    else float(effective_object_max_depenetration_velocity)
                ),
                "squeeze_mode": "fast_ramp_with_initial_object_hold",
                "squeeze_ramp_steps": int(args.physics_squeeze_ramp_steps),
                "squeeze_object_hold_steps": int(args.physics_squeeze_object_hold_steps),
                "squeeze_object_release_pen_m": (
                    float(manifest.get("gel", {}).get("recess_m", 0.0006))
                    if args.physics_squeeze_object_release_pen < 0.0
                    else float(args.physics_squeeze_object_release_pen)
                ),
                "squeeze_pen_check_after_steps": int(args.physics_squeeze_pen_check_after_steps),
                "friction_margin_model": "mu_eff * (N_left + N_right) vs object_mass * 9.81",
                "mu_eff": float(effective_contact_mu()),
                "object_weight_N": float(args.object_mass * 9.81),
                "force_spike_limit_N": float(args.physics_force_spike_limit_N),
            }
            physics_probe["pedestal"] = place_pedestal_under(physics_contact_center_np)
            place_object_at_live_pads(physics_contact_center_np)
            placement_snapshot = current_live_contact_snapshot()
            physics_probe["after_place_precheck"] = placement_snapshot
            if args.physics_calibrate_close:
                calib = physics_probe.get("close_calibration", {})
                place_row = calib.get("place_row") or {}
                preclose_row = calib.get("preclose_row") or {}
                place_pens = placement_snapshot.get("pen_stat_m") or placement_snapshot.get("pen_m", {})
                if "left" in place_pens and "right" in place_pens:
                    pen_imbalance = abs(float(place_pens["left"]) - float(place_pens["right"]))
                    balance_target = float(calib.get("live_place_depth_m") or args.live_contact_depth)
                    balance_tol = placement_balance_tolerance(balance_target)
                    physics_probe["after_place_pen_balance"] = {
                        "target_pen_m": float(balance_target),
                        "pen_imbalance_m": float(pen_imbalance),
                        "tolerance_m": float(balance_tol),
                        "pass": bool(pen_imbalance <= balance_tol),
                    }
                    if pen_imbalance > balance_tol:
                        raise RuntimeError(
                            "calibrated LSQ placement is one-sided: "
                            f"pen_stat_m={place_pens} pen_max_m={placement_snapshot.get('pen_max_m')} "
                            f"imbalance_m={pen_imbalance:.6g} "
                            f"tolerance_m={balance_tol:.6g}"
                        )
                flog(
                    "calibrated_place "
                    f"center={physics_contact_center_np.tolist()} "
                    f"place_step={place_row.get('step')} place_target={place_row.get('target')} "
                    f"refined_target={place_row.get('refined_target')} "
                    f"preclose_step={preclose_row.get('step')} preclose_target={preclose_row.get('target')} "
                    f"pred_place_pen={place_row.get('predicted_pen_m')} "
                    f"actual_place_pen_stat={placement_snapshot.get('pen_stat_m')} "
                    f"actual_place_pen_max={placement_snapshot.get('pen_max_m')}"
                )
            physics_probe["after_place"] = assert_placement_within_recess(
                "physics_probe", require_engaged=bool(args.physics_calibrate_close)
            )
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
                if args.physics_calibrate_close:
                    recess = float(manifest.get("gel", {}).get("recess_m", 0.0006))
                    squeeze_place_depth = grip_pen_target(recess)
                    raw_squeeze_center_np = object_center_from_live_pads(contact_depth=squeeze_place_depth)
                    physics_contact_center_np, squeeze_live_state, squeeze_center_balance = balance_live_contact_center(
                        raw_squeeze_center_np,
                        squeeze_place_depth,
                    )
                    physics_probe["pre_squeeze_raw_contact_center"] = raw_squeeze_center_np.tolist()
                    physics_probe["pre_squeeze_contact_center_balance"] = squeeze_center_balance
                    physics_probe["pre_squeeze_contact_center"] = physics_contact_center_np.tolist()
                    physics_probe["planned_contact_center"] = physics_contact_center_np.tolist()
                    physics_probe["pre_squeeze_contact_center_diagnostics"] = placement_center_diagnostics(
                        physics_contact_center_np,
                        squeeze_place_depth,
                    )
                    physics_probe["pre_squeeze_live_gap_state"] = squeeze_live_state
                    physics_probe["pedestal_after_close_recenter"] = place_pedestal_under(physics_contact_center_np)
                    place_object_at_live_pads(physics_contact_center_np)
                    recentered_snapshot = current_live_contact_snapshot()
                    physics_probe["after_close_recenter"] = recentered_snapshot
                    place_pens = recentered_snapshot.get("pen_stat_m") or recentered_snapshot.get("pen_m", {})
                    if "left" in place_pens and "right" in place_pens:
                        pen_imbalance = abs(float(place_pens["left"]) - float(place_pens["right"]))
                        balance_tol = placement_balance_tolerance(squeeze_place_depth)
                        physics_probe["after_close_recenter_pen_balance"] = {
                            "target_pen_m": float(squeeze_place_depth),
                            "pen_imbalance_m": float(pen_imbalance),
                            "tolerance_m": float(balance_tol),
                            "pass": bool(pen_imbalance <= balance_tol),
                        }
                        if pen_imbalance > balance_tol:
                            raise RuntimeError(
                                "pre-squeeze live recenter is one-sided: "
                                f"pen_stat_m={place_pens} pen_max_m={recentered_snapshot.get('pen_max_m')} "
                                f"imbalance_m={pen_imbalance:.6g} "
                                f"tolerance_m={balance_tol:.6g}"
                            )
                squeeze_target = squeezed_target_from(close_final_target, args.gripper_target)
                physics_probe["squeeze"] = {
                    "margin_rad": float(args.physics_squeeze_margin),
                    "close_final_target": close_final_target,
                    "squeeze_target": float(squeeze_target),
                    "ramp_steps": int(args.physics_squeeze_ramp_steps),
                    "object_hold_steps": int(args.physics_squeeze_object_hold_steps),
                    "object_hold_release_pen_m": (
                        float(manifest.get("gel", {}).get("recess_m", 0.0006))
                        if args.physics_squeeze_object_release_pen < 0.0
                        else float(args.physics_squeeze_object_release_pen)
                    ),
                }
                import torch

                squeeze_hold_pose = None
                if args.physics_squeeze_object_hold_steps != 0:
                    squeeze_hold_pose = torch.cat([obj.data.root_pos_w, obj.data.root_quat_w], dim=1).clone()
                squeeze_rows = run_gripper_target_steps(
                    squeeze_target,
                    args.physics_squeeze_settle_steps,
                    "SQUEEZE",
                    physics_trace,
                    start_value=close_final_target,
                    ramp_steps=args.physics_squeeze_ramp_steps,
                    hold_object_pose=squeeze_hold_pose,
                    hold_object_steps=args.physics_squeeze_object_hold_steps,
                    hold_release_pen=args.physics_squeeze_object_release_pen,
                    hold_live_contact_depth=squeeze_place_depth,
                )
                squeeze_gate = contact_gate_pass(current_contact_forces())
                physics_probe["squeeze"]["rows"] = squeeze_rows
                physics_probe["squeeze"]["summary"] = summarize_phase_rows(squeeze_rows)
                physics_probe["squeeze"]["object_hold_active_rows"] = int(
                    sum(1 for r in squeeze_rows if r.get("object_hold_active"))
                )
                physics_probe["squeeze"]["object_hold_released"] = bool(
                    any(r.get("object_hold_released_after_step") for r in squeeze_rows)
                )
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
                hold_summary = summarize_phase_rows(hold_rows)
                physics_probe["hold"] = {
                    "steps_requested": int(args.physics_hold_steps),
                    "steps_run": len(hold_rows),
                    "force_gate_N": float(args.physics_force_gate_N),
                    "summary": hold_summary,
                    "min_force_L_N": float(min(hold_force_L)) if hold_force_L else 0.0,
                    "min_force_R_N": float(min(hold_force_R)) if hold_force_R else 0.0,
                    "force_gate_pass": bool(
                        len(hold_rows) >= 100
                        and hold_force_L
                        and hold_force_R
                        and min(hold_force_L) >= args.physics_force_gate_N
                        and min(hold_force_R) >= args.physics_force_gate_N
                        and hold_summary.get("min_friction_margin_ratio", 0.0) >= 1.0
                        and hold_summary.get("force_spike_pass", False)
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
            lift_pen_L = [float((r.get("pen_stat_m") or r["pen_m"]).get("left", 0.0)) for r in lift_rows]
            lift_pen_R = [float((r.get("pen_stat_m") or r["pen_m"]).get("right", 0.0)) for r in lift_rows]
            lift_summary = summarize_phase_rows(lift_rows)
            physics_probe["lift"] = {
                "steps_requested": int(args.physics_lift_steps),
                "steps_run": len(lift_rows),
                "height_command_m": float(args.physics_lift_height),
                "per_step_root_delta_m": float(args.physics_lift_height / max(args.physics_lift_steps, 1)),
                "summary": lift_summary,
                "min_force_L_N": float(min(lift_force_L)) if lift_force_L else 0.0,
                "min_force_R_N": float(min(lift_force_R)) if lift_force_R else 0.0,
                "min_pen_L_m": float(min(lift_pen_L)) if lift_pen_L else 0.0,
                "min_pen_R_m": float(min(lift_pen_R)) if lift_pen_R else 0.0,
            }
            physics_probe["contact_force"] = contact_force_info
            physics_probe["passive_joint_names"] = passive_log_names
            physics_probe["trace"] = physics_trace
            if args.physics_calibrate_close:
                physics_probe["grip_pen_band"] = assert_trace_pen_within_grip_band(
                    "physics_probe", physics_trace, raise_on_failure=False
                )
            physics_probe["round3_gate_pass"] = bool(
                physics_probe.get("hold", {}).get("force_gate_pass", False)
                and len(lift_rows) >= int(args.physics_lift_steps)
                and physics_probe["lift"]["min_force_L_N"] > 0.0
                and physics_probe["lift"]["min_force_R_N"] > 0.0
                and lift_summary.get("min_friction_margin_ratio", 0.0) >= 1.0
                and lift_summary.get("force_spike_pass", False)
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
        if "inline_overlay_timing" in locals() and inline_overlay_timing:
            flog(f"TACTILE_OVERLAY_TIMING mean_render_ms={np.mean(inline_overlay_timing) * 1e3:.3f} "
                 f"max_render_ms={np.max(inline_overlay_timing) * 1e3:.3f} updates={len(inline_overlay_timing)}")
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
            "runtime_live_pen_stat_m": (
                {} if pseudo_pen else {
                    side: float(_pen_grid_stats(_pen_for_object(
                        args.object,
                        args.object_size,
                        np.asarray(T),
                        gel_grid(32, 0.009),
                    ))["stat_m"])
                    for side, T in live_T_gel_obj.items()
                }
            ),
            "physics_probe": physics_probe,
            "world_camera": world_camera_info,
            "livestream": livestream_info,
            "pseudo_jaw_probe": pseudo_probe,
            "gripper_scan_probe": gripper_scan,
            "_tactile_overlay": tactile_overlay,
        }
    except Exception as exc:
        flog(f"ISAAC_SCENE_FAIL {type(exc).__name__}: {exc}")
        if isinstance(physics_probe, dict):
            physics_probe["scene_setup_failure"] = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "stage": (
                    "ik_dynamic_grasp"
                    if getattr(args, "grasp_mode", "legacy") == "ik"
                    else "legacy_scene_setup"
                ),
            }
            if "trace" not in physics_probe:
                physics_probe["trace"] = []
        return {
            "isaac_scene": False,
            "isaac_error": f"{type(exc).__name__}: {exc}",
            "physics_probe": physics_probe,
            "livestream": livestream_info,
        }


def run_demo(args) -> dict:
    Path(PROGRESS).write_text("")
    if args.object_at_pads and args.live_contact_depth > args.max_live_contact_depth:
        raise SystemExit(
            f"--live-contact-depth={args.live_contact_depth:.6g} exceeds "
            f"--max-live-contact-depth={args.max_live_contact_depth:.6g}; "
            "keep placement within the configured safe grip depth and close the gripper dynamically."
        )
    tag = args.tag or f"{args.mode}_{int(time.time())}"
    out_dir = Path("/work/runs/sim_grasp") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    args._out_dir = str(out_dir)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    physics_pick_probe_requested = bool(args.mode == "pick" and args.physics_lift_probe)

    def write_failure_sequence_npz(reason: str, isaac_info: dict) -> None:
        physics_probe_npz = isaac_info.get("physics_probe", {}) if isinstance(isaac_info, dict) else {}
        close_calib = (
            physics_probe_npz.get("close_calibration", {})
            if isinstance(physics_probe_npz, dict)
            else {}
        )
        calib_rows = close_calib.get("rows", []) if isinstance(close_calib, dict) else []

        def row_float(row, key):
            value = row.get(key)
            return np.nan if value is None else float(value)

        def physics_failure_rows():
            trace = physics_probe_npz.get("trace", []) if isinstance(physics_probe_npz, dict) else []
            if trace:
                return trace
            rows = []
            setup_failure = (
                physics_probe_npz.get("scene_setup_failure")
                if isinstance(physics_probe_npz, dict)
                else None
            )
            if isinstance(setup_failure, dict):
                rows.append({
                    "phase": "FAIL_SCENE_SETUP",
                    "target": np.nan,
                    "pen_m": {},
                    "pen_stat_m": {},
                    "pen_max_m": {},
                    "force_N": {"left": np.nan, "right": np.nan},
                    "friction_margin": {},
                    "object_pos_w": [np.nan, np.nan, np.nan],
                    "pad_pose_w": {},
                    "gel_origin_w": {},
                    "grip_symmetry": {},
                    "robot_root_z_m": np.nan,
                    "passive_joint_pos": [],
                    "object_hold_active": False,
                    "scene_setup_failure": setup_failure,
                })
            for key, phase in (
                ("after_place_precheck", "FAIL_AFTER_PLACE_PRECHECK"),
                ("after_place", "FAIL_AFTER_PLACE"),
                ("after_close_recenter", "FAIL_AFTER_CLOSE_RECENTER"),
                ("before_lift", "FAIL_BEFORE_LIFT"),
            ):
                snap = physics_probe_npz.get(key) if isinstance(physics_probe_npz, dict) else None
                if not isinstance(snap, dict):
                    continue
                obj_pose = snap.get("object_pose_w", {}) if isinstance(snap.get("object_pose_w"), dict) else {}
                rows.append({
                    "phase": phase,
                    "target": np.nan,
                    "pen_m": snap.get("pen_m", {}),
                    "pen_stat_m": snap.get("pen_stat_m", snap.get("pen_m", {})),
                    "pen_max_m": snap.get("pen_max_m", snap.get("pen_m", {})),
                    "force_N": {"left": np.nan, "right": np.nan},
                    "friction_margin": {},
                    "object_pos_w": obj_pose.get("pos", [np.nan, np.nan, np.nan]),
                    "pad_pose_w": snap.get("pad_pose_w", {}),
                    "gel_origin_w": snap.get("gel_origin_w", {}),
                    "grip_symmetry": snap.get("grip_symmetry", {}),
                    "robot_root_z_m": np.nan,
                    "passive_joint_pos": [],
                    "object_hold_active": False,
                })
            return rows

        failure_physics_rows = physics_failure_rows()

        def row_pen(row, side):
            return float(row.get("pen_m", {}).get(side, np.nan))

        def row_pen_stat(row, side):
            return float((row.get("pen_stat_m") or row.get("pen_m", {})).get(side, np.nan))

        def row_pen_max(row, side):
            return float((row.get("pen_max_m") or row.get("pen_m", {})).get(side, np.nan))

        def row_force(row, side):
            return float(row.get("force_N", {}).get(side, np.nan))

        def row_vec(row, key, default):
            value = row.get(key, default)
            if value is None:
                return default
            if isinstance(value, (list, tuple)) and len(value) == len(default):
                return value
            return default

        def nested_vec(row, outer, side, key, default):
            value = row.get(outer, {}).get(side, {}).get(key, default)
            if value is None:
                return default
            if isinstance(value, (list, tuple)) and len(value) == len(default):
                return value
            return default

        def nested_side_vec(row, outer, side, default):
            value = row.get(outer, {}).get(side, default)
            if value is None:
                return default
            if isinstance(value, (list, tuple)) and len(value) == len(default):
                return value
            return default

        def symmetry_vec(row, key, default):
            value = row.get("grip_symmetry", {}).get(key, default)
            if value is None:
                return default
            if isinstance(value, (list, tuple)) and len(value) == len(default):
                return value
            return default

        physics_object_pos = np.asarray([
            row_vec(r, "object_pos_w", [np.nan, np.nan, np.nan]) for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_ee_pos = np.asarray([
            row_vec(r.get("ee_pose_w", {}), "pos", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_ee_target_pos = np.asarray([
            row_vec(r.get("ee_target_pose_w", {}), "pos", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_pad_pos_L = np.asarray([
            nested_vec(r, "pad_pose_w", "left", "pos", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_pad_pos_R = np.asarray([
            nested_vec(r, "pad_pose_w", "right", "pos", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_pad_quat_L = np.asarray([
            nested_vec(r, "pad_pose_w", "left", "quat_wxyz", [np.nan, np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 4), dtype=np.float32)
        physics_pad_quat_R = np.asarray([
            nested_vec(r, "pad_pose_w", "right", "quat_wxyz", [np.nan, np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 4), dtype=np.float32)
        physics_gel_origin_L = np.asarray([
            nested_side_vec(r, "gel_origin_w", "left", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_gel_origin_R = np.asarray([
            nested_side_vec(r, "gel_origin_w", "right", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_grip_axis = np.asarray([
            symmetry_vec(r, "axis_w", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_gel_midpoint = np.asarray([
            symmetry_vec(r, "gel_midpoint_w", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_object_minus_gel_midpoint = np.asarray([
            symmetry_vec(r, "object_minus_gel_midpoint_w", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_passive_joint_pos = np.asarray([
            r.get("passive_joint_pos") or [np.nan, np.nan, np.nan, np.nan]
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 4), dtype=np.float32)
        physics_ik_pos_error = np.asarray([
            r.get("ik_error", {}).get("pos_error_w", [np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 3), dtype=np.float32)
        physics_arm_joint_pos = np.asarray([
            r.get("arm_joint_pos", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 6), dtype=np.float32)
        physics_arm_joint_target = np.asarray([
            r.get("ik_diagnostics", {}).get("arm_joint_target", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            for r in failure_physics_rows
        ], dtype=np.float32) if failure_physics_rows else np.zeros((0, 6), dtype=np.float32)

        np.savez_compressed(
            out_dir / "sequence.npz",
            phase=np.asarray([], dtype="<U1"),
            failure_reason=np.asarray([reason]),
            physics_phase=np.asarray([r.get("phase", "") for r in failure_physics_rows]),
            contact_force_N_L=np.asarray([row_force(r, "left") for r in failure_physics_rows], dtype=np.float32),
            contact_force_N_R=np.asarray([row_force(r, "right") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_L=np.asarray([row_pen(r, "left") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_R=np.asarray([row_pen(r, "right") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_stat_L=np.asarray([row_pen_stat(r, "left") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_stat_R=np.asarray([row_pen_stat(r, "right") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_max_L=np.asarray([row_pen_max(r, "left") for r in failure_physics_rows], dtype=np.float32),
            physics_pen_max_R=np.asarray([row_pen_max(r, "right") for r in failure_physics_rows], dtype=np.float32),
            physics_object_pos_w=physics_object_pos,
            physics_ee_pos_w=physics_ee_pos,
            physics_ee_target_pos_w=physics_ee_target_pos,
            physics_ik_pos_error_w=physics_ik_pos_error,
            physics_ik_z_error_m=np.asarray([
                float(r.get("ik_error", {}).get("z_error_m", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_ik_pos_error_norm_m=np.asarray([
                float(r.get("ik_error", {}).get("pos_error_norm_m", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_ik_orientation_error_rad=np.asarray([
                float(r.get("ik_error", {}).get("orientation_error_rad", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_ik_jacobian_rank=np.asarray([
                int(r.get("ik_diagnostics", {}).get("jacobian", {}).get("rank", -1))
                for r in failure_physics_rows
            ], dtype=np.int32),
            physics_ik_jacobian_condition=np.asarray([
                row_float(r.get("ik_diagnostics", {}).get("jacobian", {}), "condition")
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_arm_joint_pos=physics_arm_joint_pos,
            physics_arm_joint_target=physics_arm_joint_target,
            physics_pad_pos_w_L=physics_pad_pos_L,
            physics_pad_pos_w_R=physics_pad_pos_R,
            physics_pad_quat_wxyz_L=physics_pad_quat_L,
            physics_pad_quat_wxyz_R=physics_pad_quat_R,
            physics_gel_origin_w_L=physics_gel_origin_L,
            physics_gel_origin_w_R=physics_gel_origin_R,
            physics_grip_axis_w=physics_grip_axis,
            physics_gel_midpoint_w=physics_gel_midpoint,
            physics_object_minus_gel_midpoint_w=physics_object_minus_gel_midpoint,
            physics_object_axial_offset_m=np.asarray([
                float(r.get("grip_symmetry", {}).get("object_axial_offset_m", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_gel_origin_separation_m=np.asarray([
                float(r.get("grip_symmetry", {}).get("gel_origin_separation_m", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_pad_separation_m=np.asarray([
                float(r.get("grip_symmetry", {}).get("pad_separation_m", np.nan))
                for r in failure_physics_rows
            ], dtype=np.float32),
            physics_object_hold_active=np.asarray([
                bool(r.get("object_hold_active", False)) for r in failure_physics_rows
            ], dtype=np.bool_),
            physics_robot_root_z_m=np.asarray([
                float(r.get("robot_root_z_m", np.nan)) for r in failure_physics_rows
            ], dtype=np.float32),
            passive_joint_pos=physics_passive_joint_pos,
            close_calibration_step=np.asarray(
                [int(r.get("step", 0)) for r in calib_rows], dtype=np.int32
            ),
            close_calibration_target=np.asarray(
                [row_float(r, "target") for r in calib_rows], dtype=np.float32
            ),
            close_calibration_gap_m=np.asarray(
                [row_float(r, "gap_m") for r in calib_rows], dtype=np.float32
            ),
            close_calibration_required_gap_m=np.asarray(
                [row_float(r, "placement_required_gap_m") for r in calib_rows], dtype=np.float32
            ),
            close_calibration_support_required_gap_m=np.asarray(
                [row_float(r, "placement_support_required_gap_m") for r in calib_rows],
                dtype=np.float32,
            ),
            close_calibration_gap_clearance_m=np.asarray(
                [row_float(r, "placement_gap_clearance_m") for r in calib_rows],
                dtype=np.float32,
            ),
            close_calibration_predicted_pen_error_m=np.asarray(
                [row_float(r, "placement_predicted_pen_error_m") for r in calib_rows],
                dtype=np.float32,
            ),
        )

    def abort_demo(reason: str, isaac_info: dict = None) -> None:
        isaac_info = isaac_info or {}

        def json_safe(value):
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            if isinstance(value, np.ndarray):
                return json_safe(value.tolist())
            if isinstance(value, np.generic):
                return json_safe(value.item())
            if isinstance(value, float):
                return value if math.isfinite(value) else None
            return value

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
        try:
            write_failure_sequence_npz(reason, isaac_info)
        except Exception as npz_exc:
            summary["failure_sequence_npz_error"] = f"{type(npz_exc).__name__}: {npz_exc}"
            np.savez_compressed(
                out_dir / "sequence.npz",
                phase=np.asarray([], dtype="<U1"),
                physics_phase=np.asarray([], dtype="<U1"),
                failure_reason=np.asarray([reason]),
                failure_sequence_npz_error=np.asarray([summary["failure_sequence_npz_error"]]),
            )
        (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
        flog(f"GRASP_DEMO_FAIL {reason}")
        print("GRASP_DEMO_FAIL", out_dir / "summary.json", flush=True)
        os._exit(1)

    isaac_info = _try_start_isaac(str(manifest_path), args) if args.start_isaac else {"isaac_scene": False}
    # Private runtime handles must never enter sequence/summary schemas.
    tactile_overlay = isaac_info.pop("_tactile_overlay", None)
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
    def float_or_nan(value):
        return np.nan if value is None else float(value)

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
    physics_pen_stat_L = np.asarray([
        float((r.get("pen_stat_m") or r.get("pen_m", {})).get("left", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_pen_stat_R = np.asarray([
        float((r.get("pen_stat_m") or r.get("pen_m", {})).get("right", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_pen_max_L = np.asarray([
        float((r.get("pen_max_m") or r.get("pen_m", {})).get("left", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_pen_max_R = np.asarray([
        float((r.get("pen_max_m") or r.get("pen_m", {})).get("right", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_friction_margin_N = np.asarray([
        float(r.get("friction_margin", {}).get("margin_N", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_friction_margin_ratio = np.asarray([
        float(r.get("friction_margin", {}).get("margin_ratio", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_friction_capacity_N = np.asarray([
        float(r.get("friction_margin", {}).get("friction_capacity_N", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_object_pos = np.asarray([
        r.get("object_pos_w", [np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_ee_pos = np.asarray([
        r.get("ee_pose_w", {}).get("pos", [np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_ee_target_pos = np.asarray([
        r.get("ee_target_pose_w", {}).get("pos", [np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_ik_pos_error = np.asarray([
        r.get("ik_error", {}).get("pos_error_w", [np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_ik_z_error = np.asarray([
        float(r.get("ik_error", {}).get("z_error_m", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_ik_pos_error_norm = np.asarray([
        float(r.get("ik_error", {}).get("pos_error_norm_m", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_ik_orientation_error = np.asarray([
        float(r.get("ik_error", {}).get("orientation_error_rad", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_ik_jacobian_rank = np.asarray([
        int(r.get("ik_diagnostics", {}).get("jacobian", {}).get("rank", -1)) for r in physics_trace
    ], dtype=np.int32)
    physics_ik_jacobian_condition = np.asarray([
        float_or_nan(r.get("ik_diagnostics", {}).get("jacobian", {}).get("condition", np.nan)) for r in physics_trace
    ], dtype=np.float32)
    physics_arm_joint_pos = np.asarray([
        r.get("arm_joint_pos", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]) for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 6), dtype=np.float32)
    physics_arm_joint_target = np.asarray([
        r.get("ik_diagnostics", {}).get("arm_joint_target", [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 6), dtype=np.float32)
    physics_pad_pos_L = np.asarray([
        r.get("pad_pose_w", {}).get("left", {}).get("pos", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_pad_pos_R = np.asarray([
        r.get("pad_pose_w", {}).get("right", {}).get("pos", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_pad_quat_L = np.asarray([
        r.get("pad_pose_w", {}).get("left", {}).get("quat_wxyz", [np.nan, np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 4), dtype=np.float32)
    physics_pad_quat_R = np.asarray([
        r.get("pad_pose_w", {}).get("right", {}).get("quat_wxyz", [np.nan, np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 4), dtype=np.float32)
    physics_gel_origin_L = np.asarray([
        r.get("gel_origin_w", {}).get("left", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_gel_origin_R = np.asarray([
        r.get("gel_origin_w", {}).get("right", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_grip_axis = np.asarray([
        r.get("grip_symmetry", {}).get("axis_w", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_gel_midpoint = np.asarray([
        r.get("grip_symmetry", {}).get("gel_midpoint_w", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_object_minus_gel_midpoint = np.asarray([
        r.get("grip_symmetry", {}).get("object_minus_gel_midpoint_w", [np.nan, np.nan, np.nan])
        for r in physics_trace
    ], dtype=np.float32) if physics_trace else np.zeros((0, 3), dtype=np.float32)
    physics_object_axial_offset = np.asarray([
        float_or_nan(r.get("grip_symmetry", {}).get("object_axial_offset_m", np.nan))
        for r in physics_trace
    ], dtype=np.float32)
    physics_gel_origin_separation = np.asarray([
        float_or_nan(r.get("grip_symmetry", {}).get("gel_origin_separation_m", np.nan))
        for r in physics_trace
    ], dtype=np.float32)
    physics_pad_separation = np.asarray([
        float_or_nan(r.get("grip_symmetry", {}).get("pad_separation_m", np.nan))
        for r in physics_trace
    ], dtype=np.float32)
    physics_object_hold_active = np.asarray([
        bool(r.get("object_hold_active", False)) for r in physics_trace
    ], dtype=np.bool_)
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
        physics_pen_stat_L=physics_pen_stat_L,
        physics_pen_stat_R=physics_pen_stat_R,
        physics_pen_max_L=physics_pen_max_L,
        physics_pen_max_R=physics_pen_max_R,
        physics_friction_margin_N=physics_friction_margin_N,
        physics_friction_margin_ratio=physics_friction_margin_ratio,
        physics_friction_capacity_N=physics_friction_capacity_N,
        physics_object_pos_w=physics_object_pos,
        physics_ee_pos_w=physics_ee_pos,
        physics_ee_target_pos_w=physics_ee_target_pos,
        physics_ik_pos_error_w=physics_ik_pos_error,
        physics_ik_z_error_m=physics_ik_z_error,
        physics_ik_pos_error_norm_m=physics_ik_pos_error_norm,
        physics_ik_orientation_error_rad=physics_ik_orientation_error,
        physics_ik_jacobian_rank=physics_ik_jacobian_rank,
        physics_ik_jacobian_condition=physics_ik_jacobian_condition,
        physics_arm_joint_pos=physics_arm_joint_pos,
        physics_arm_joint_target=physics_arm_joint_target,
        physics_pad_pos_w_L=physics_pad_pos_L,
        physics_pad_pos_w_R=physics_pad_pos_R,
        physics_pad_quat_wxyz_L=physics_pad_quat_L,
        physics_pad_quat_wxyz_R=physics_pad_quat_R,
        physics_gel_origin_w_L=physics_gel_origin_L,
        physics_gel_origin_w_R=physics_gel_origin_R,
        physics_grip_axis_w=physics_grip_axis,
        physics_gel_midpoint_w=physics_gel_midpoint,
        physics_object_minus_gel_midpoint_w=physics_object_minus_gel_midpoint,
        physics_object_axial_offset_m=physics_object_axial_offset,
        physics_gel_origin_separation_m=physics_gel_origin_separation,
        physics_pad_separation_m=physics_pad_separation,
        physics_object_hold_active=physics_object_hold_active,
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
    montage_info = {"enabled": False}
    if args.world_camera and args.make_montage:
        try:
            from novbts.simulation.runtime.grasp_montage import make_grasp_montage

            montage_info = make_grasp_montage(
                out_dir,
                world_frames=isaac_info.get("world_camera", {}).get("frames", []),
                tactile_every=args.tactile_every,
            )
            montage_info["enabled"] = True
        except Exception as montage_exc:
            montage_info = {
                "enabled": False,
                "error": f"{type(montage_exc).__name__}: {montage_exc}",
            }
            flog(f"MONTAGE_FAIL {montage_info['error']}")
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
    livestream_hold_info = {"enabled": False, "reason": "not requested"}
    if int(getattr(args, "livestream", 0)) == 1 and marker in ("GRASP_DEMO_OK", "PRESS_OK"):
        livestream_hold_info = hold_livestream(args, _LIVE_SIMULATION_APP)
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
        "montage": montage_info,
        "livestream_hold": livestream_hold_info,
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
    args = build_arg_parser().parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
