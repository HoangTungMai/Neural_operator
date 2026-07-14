#!/usr/bin/env python3
"""Standalone one-pad VBTS sandbox.

``--runtime host --drive script`` is deliberately Isaac-free and exercises the
same analytic penetration, shear tracker, frozen FNO and output contract as the
interactive Isaac viewport.  Isaac is imported only by ``run_interactive``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# The Isaac image executes this file directly via /isaac-sim/python.sh, while
# host mode normally uses ``python -m`` with PYTHONPATH=src.  Bootstrap both
# layouts before importing project modules.
if "/work/src" not in sys.path:
    sys.path.append("/work/src")
for _path in glob.glob("/workspace/isaaclab/source/*"):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from novbts.paths import ROOT, RUNS, ensure
from novbts.simulation.runtime.fno_export import DEFAULT_OUT as DEFAULT_CKPT
from novbts.simulation.runtime.gel_contact import ShearTracker, gel_grid, pose_to_T
from novbts.simulation.runtime.grasp_tactile_util import pen_for_object
from novbts.simulation.runtime.vbts_sensor import VBTSSensor


def progress_path() -> Path:
    return Path(os.environ.get("SENSOR_PROGRESS", str(ROOT / "sim_sensor_progress.txt")))


def flog(message: str) -> None:
    path = progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        handle.flush()


def _save_gray_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


@dataclass
class SensorState:
    x: float
    y: float
    z: float


class SensorController:
    """Single source of truth shared by script, keyboard and UI controls."""

    gel_half_m = 0.009
    max_pen_m = 0.00075
    clearance_m = 0.002

    def __init__(
        self,
        *,
        object_size: float,
        on_change: Callable[[str, float], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        self.object_size = float(object_size)
        self.half_extent = self.object_size / 2.0
        self.x_min = -(self.gel_half_m - self.half_extent)
        self.x_max = self.gel_half_m - self.half_extent
        if self.x_max <= 0.0:
            raise ValueError("object-size must be smaller than the 18 mm tactile grid span")
        self.y_min, self.y_max = self.x_min, self.x_max
        self.z_min = self.half_extent - self.max_pen_m
        self.z_max = self.half_extent + self.clearance_m
        self.on_change = on_change
        self.on_reset = on_reset
        self.state = SensorState(0.0, 0.0, self.z_max)

    def bounds(self, axis: str) -> tuple[float, float]:
        return getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")

    def set_axis(self, axis: str, value: float) -> float:
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"unknown axis {axis!r}")
        lo, hi = self.bounds(axis)
        clamped = float(np.clip(float(value), lo, hi))
        if float(getattr(self.state, axis)) != clamped:
            setattr(self.state, axis, clamped)
            if self.on_change is not None:
                self.on_change(axis, clamped)
        return clamped

    def set_pose(self, x: float, y: float, z: float) -> None:
        self.set_axis("x", x)
        self.set_axis("y", y)
        self.set_axis("z", z)

    def reset(self) -> None:
        self.set_pose(0.0, 0.0, self.z_max)
        if self.on_reset is not None:
            self.on_reset()


def iter_script_targets(controller: SensorController) -> Iterator[tuple[str, float, float, float]]:
    """Yield the deterministic 105-frame press/slide/release trajectory."""
    rest_z = controller.z_max
    press_z = controller.half_extent - 0.0006
    for _ in range(10):
        yield "rest", 0.0, 0.0, rest_z
    for z in np.linspace(rest_z, press_z, 20):
        yield "press", 0.0, 0.0, float(z)
    for _ in range(10):
        yield "hold", 0.0, 0.0, press_z
    for x in np.linspace(0.0, 0.002, 20):
        yield "slide", float(x), 0.0, press_z
    for x in np.linspace(0.002, 0.0, 20):
        yield "slide", float(x), 0.0, press_z
    for z in np.linspace(press_z, rest_z, 20):
        yield "release", 0.0, 0.0, float(z)
    for _ in range(5):
        yield "reset", 0.0, 0.0, rest_z


def render_frame(args, sensor: VBTSSensor, tracker: ShearTracker, controller: SensorController, grid: np.ndarray) -> dict:
    state = controller.state
    transform = pose_to_T((state.x, state.y, state.z), (1.0, 0.0, 0.0, 0.0))
    pen = pen_for_object(args.object, args.object_size, transform, grid)
    shear = tracker.update(transform, pen)
    started = time.perf_counter()
    rendered = sensor.render(pen, float(shear[0]), float(shear[1]), mu=args.mu, E=args.E, noise=not args.no_noise)
    elapsed = time.perf_counter() - started
    return {
        "image": rendered["img"][0, 0].detach().cpu().numpy(),
        "flow": rendered["flow"][0].detach().cpu().numpy(),
        "pen": pen,
        "pen_max": float(np.max(pen)),
        "shear": np.asarray(shear, dtype=np.float32),
        "pose": np.asarray((state.x, state.y, state.z), dtype=np.float32),
        "timing_s": elapsed,
    }


class ScriptRecorder:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.flows: list[np.ndarray] = []
        self.pen_max: list[float] = []
        self.shear: list[np.ndarray] = []
        self.pose: list[np.ndarray] = []
        self.phase: list[str] = []
        self.timings_s: list[float] = []

    def append(self, phase: str, frame: dict) -> None:
        self.images.append(np.asarray(frame["image"], dtype=np.float32))
        self.flows.append(np.asarray(frame["flow"], dtype=np.float32))
        self.pen_max.append(float(frame["pen_max"]))
        self.shear.append(np.asarray(frame["shear"], dtype=np.float32))
        self.pose.append(np.asarray(frame["pose"], dtype=np.float32))
        self.phase.append(str(phase))
        self.timings_s.append(float(frame["timing_s"]))

    def write(self, out_dir: Path) -> dict:
        ensure(out_dir)
        images = np.stack(self.images).astype(np.float32)
        flow = np.stack(self.flows).astype(np.float32)
        pen_max = np.asarray(self.pen_max, dtype=np.float32)
        shear = np.stack(self.shear).astype(np.float32)
        pose = np.stack(self.pose).astype(np.float32)
        phase = np.asarray(self.phase, dtype="<U8")
        timings_s = np.asarray(self.timings_s, dtype=np.float32)
        npz_path = out_dir / "sensor_sequence.npz"
        np.savez_compressed(
            npz_path,
            images=images,
            pen_max=pen_max,
            shear=shear,
            pose=pose,
            flow=flow,
            phase=phase,
            timings_s=timings_s,
        )
        indices = {
            "rest": int(np.flatnonzero(phase == "rest")[0]),
            "pressed": int(np.flatnonzero(phase == "press")[-1]),
            "shear": int(np.linalg.norm(shear, axis=1).argmax()),
            "released": int(np.flatnonzero(phase == "release")[-1]),
        }
        for label, index in indices.items():
            _save_gray_png(images[index], out_dir / f"tactile_{label}.png")
        report = {
            "frames": int(images.shape[0]),
            "npz": str(npz_path),
            "snapshot_indices": indices,
            "max_pen_m": float(pen_max.max()),
            "max_shear_m": float(np.linalg.norm(shear, axis=1).max()),
            "mean_render_ms": float(timings_s.mean() * 1e3),
        }
        (out_dir / "sensor_report.json").write_text(json.dumps(report, indent=2))
        return report


def _out_dir(args) -> Path:
    return Path(args.out) if args.out else RUNS / "sim_sensor" / args.tag


def run_script(args) -> dict:
    """Run the deterministic host-side acceptance sequence without Isaac."""
    controller = SensorController(object_size=args.object_size)
    tracker = ShearTracker(mu=args.mu)
    controller.on_reset = tracker.reset
    sensor = VBTSSensor(args.ckpt, px=args.px, sensor_side=args.sensor_side)
    grid = gel_grid(sensor.side, 0.009)
    recorder = ScriptRecorder()
    for phase, x, y, z in iter_script_targets(controller):
        if phase == "reset":
            controller.reset()
        else:
            controller.set_pose(x, y, z)
        recorder.append(phase, render_frame(args, sensor, tracker, controller, grid))
    report = recorder.write(_out_dir(args))
    flog(f"SENSOR_DEMO_OK out={_out_dir(args)} frames={report['frames']}")
    return report


def _bootstrap_isaac_paths() -> None:
    if "/work/src" not in sys.path:
        sys.path.append("/work/src")
    for path in glob.glob("/workspace/isaaclab/source/*"):
        if path not in sys.path:
            sys.path.insert(0, path)


def _frame_viewport_on_gel() -> None:
    try:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except Exception:
            from omni.isaac.core.utils.viewports import set_camera_view
        eye = [0.040, -0.040, 0.030]
        target = [0.0, 0.0, 0.001]
        set_camera_view(eye=eye, target=target)
        flog("LIVESTREAM_VIEWPORT_CAMERA_SET eye=(0.040,-0.040,0.030) target=(0.000,0.000,0.001)")
    except Exception as exc:
        flog(f"LIVESTREAM_VIEWPORT_CAMERA_FAIL {type(exc).__name__}: {exc}")


def _spawn_stage(args):
    """Create display-only USD geometry; no rigid-body or collider APIs are applied."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    gel = UsdGeom.Cube.Define(stage, "/World/Gel")
    gel.GetSizeAttr().Set(1.0)
    # USD xform ops are composed in their authored order.  Translate first so
    # the 3 mm cube is centred at -1.5 mm and its tactile top surface is z=0.
    # Authoring scale first accidentally scaled the translation itself, lifting
    # the rendered block ~1.5 mm above the analytic contact surface.
    gel.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.0015))
    gel.AddScaleOp().Set(Gf.Vec3f(0.020, 0.020, 0.003))
    gel.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.75, 0.85)])
    if args.object == "sphere":
        obj = UsdGeom.Sphere.Define(stage, "/World/Object")
        obj.GetRadiusAttr().Set(float(args.object_size / 2.0))
    elif args.object == "cube":
        obj = UsdGeom.Cube.Define(stage, "/World/Object")
        obj.GetSizeAttr().Set(1.0)
        obj.AddScaleOp().Set(Gf.Vec3f(args.object_size, args.object_size, args.object_size))
    else:
        obj = UsdGeom.Cylinder.Define(stage, "/World/Object")
        obj.GetRadiusAttr().Set(float(args.object_size / 2.0))
        obj.GetHeightAttr().Set(float(args.object_size))
    translate_op = obj.AddTranslateOp()
    obj.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.42, 0.18)])
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(750.0)
    robot = stage.GetPrimAtPath("/World/Robot")
    gel_prim = stage.GetPrimAtPath("/World/Gel")
    object_prim = stage.GetPrimAtPath("/World/Object")
    if robot.IsValid() or not gel_prim.IsValid() or not object_prim.IsValid():
        raise RuntimeError("unexpected standalone sensor stage topology")
    if gel_prim.HasAPI(UsdPhysics.RigidBodyAPI) or object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("standalone sensor geometry unexpectedly has RigidBodyAPI")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    gel_z_top = float(bbox_cache.ComputeWorldBound(gel_prim).ComputeAlignedRange().GetMax()[2])
    if abs(gel_z_top) > 1.0e-6:
        raise RuntimeError(f"rendered gel top must coincide with tactile z=0, got {gel_z_top:.9f}")
    flog("STAGE_NO_ROBOT_PRIM ok")
    flog(f"STAGE_GEL_SURFACE_OK top_z_m={gel_z_top:.9f}")
    return translate_op


def _subscribe_keyboard(controller: SensorController, step_m: float) -> tuple[object | None, dict[str, bool]]:
    """Best-effort keyboard handling; UI remains usable if transport input fails."""
    running = {"value": True}
    try:
        import carb.input
        import omni.appwindow

        window = omni.appwindow.get_default_app_window()
        keyboard = None if window is None else window.get_keyboard()
        if keyboard is None:
            raise RuntimeError("default keyboard unavailable")
        interface = carb.input.acquire_input_interface()
        event_types = (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT)
        keymap = {
            carb.input.KeyboardInput.A: ("x", -step_m, "A"),
            carb.input.KeyboardInput.D: ("x", step_m, "D"),
            carb.input.KeyboardInput.S: ("y", -step_m, "S"),
            carb.input.KeyboardInput.W: ("y", step_m, "W"),
            carb.input.KeyboardInput.Q: ("z", -step_m, "Q"),
            carb.input.KeyboardInput.E: ("z", step_m, "E"),
        }

        def on_key(event, *_unused):
            if event.type not in event_types:
                return True
            if event.input == carb.input.KeyboardInput.R:
                controller.reset()
                flog("KEYBOARD_EVENT_RECEIVED key=R")
                return True
            if hasattr(carb.input.KeyboardInput, "ESCAPE") and event.input == carb.input.KeyboardInput.ESCAPE:
                running["value"] = False
                flog("KEYBOARD_EVENT_RECEIVED key=ESCAPE")
                return True
            action = keymap.get(event.input)
            if action is not None:
                axis, delta, name = action
                controller.set_axis(axis, getattr(controller.state, axis) + delta)
                flog(f"KEYBOARD_EVENT_RECEIVED key={name}")
            return True

        subscription = interface.subscribe_to_keyboard_events(keyboard, on_key)
        flog("KEYBOARD_HANDLER_READY ok")
        return subscription, running
    except Exception as exc:
        flog(f"KEYBOARD_UNAVAILABLE {type(exc).__name__}: {exc}")
        return None, running


def run_interactive(args) -> dict:
    _bootstrap_isaac_paths()
    import inspect
    from isaaclab.app import AppLauncher

    launcher_kwargs = {"headless": True, "enable_cameras": True}
    if args.livestream:
        signature = inspect.signature(AppLauncher.__init__)
        if "livestream" not in signature.parameters and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        ):
            raise RuntimeError(f"AppLauncher does not expose livestream: {signature}")
        launcher_kwargs["livestream"] = int(args.livestream)
        if args.livestream == 1:
            present = "--/exts/omni.renderer.core/present/enabled=true"
            if present not in sys.argv:
                sys.argv.append(present)
    app = AppLauncher(**launcher_kwargs).app
    panel = None
    subscription = None
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext
        from pxr import Gf

        sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
        controller = SensorController(object_size=args.object_size)
        tracker = ShearTracker(mu=args.mu)
        controller.on_reset = tracker.reset
        translate_op = _spawn_stage(args)
        sim.reset()
        sensor = VBTSSensor(args.ckpt, px=args.px, sensor_side=args.sensor_side, device="cuda")
        grid = gel_grid(sensor.side, 0.009)
        initial = render_frame(args, sensor, tracker, controller, grid)
        from novbts.simulation.runtime.sensor_overlay import SensorControlPanel

        panel = SensorControlPanel(controller, log=flog, res=args.px)
        controller.on_change = panel.sync_axis
        if not panel.setup(initial["image"]):
            raise RuntimeError("SensorControlPanel setup failed")
        _frame_viewport_on_gel()
        subscription, running = _subscribe_keyboard(controller, args.step_mm * 1e-3)
        recorder = ScriptRecorder() if args.drive == "script" else None
        targets = iter_script_targets(controller) if args.drive == "script" else None
        while running["value"] and app.is_running():
            if targets is not None:
                try:
                    phase, x, y, z = next(targets)
                except StopIteration:
                    break
                if phase == "reset":
                    controller.reset()
                else:
                    controller.set_pose(x, y, z)
            else:
                phase = "manual"
            state = controller.state
            translate_op.Set(Gf.Vec3d(state.x, state.y, state.z))
            frame = render_frame(args, sensor, tracker, controller, grid)
            panel.update(frame["image"], frame["pen_max"], frame["shear"])
            if recorder is not None:
                recorder.append(phase, frame)
            sim.step(render=True)
        if recorder is not None:
            report = recorder.write(_out_dir(args))
            flog(f"SENSOR_DEMO_OK out={_out_dir(args)} frames={report['frames']}")
            return report
        flog("SENSOR_INTERACTIVE_EXIT")
        return {"interactive": True}
    finally:
        if subscription is not None:
            try:
                subscription.unsubscribe()
            except Exception:
                pass
        if panel is not None:
            panel.teardown()
        # Match grasp_demo's lifecycle: explicit ``SimulationApp.close()`` can
        # block indefinitely after an omni.ui window has been torn down in this
        # Isaac 4.5 image.  Let Kit shut down with the Python process instead.


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("host", "isaac"), default="host")
    parser.add_argument("--drive", choices=("script", "manual"), default="script")
    parser.add_argument("--object", choices=("sphere", "cube", "cylinder"), default="sphere")
    parser.add_argument("--object-size", type=float, default=0.008)
    parser.add_argument("--livestream", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--E", type=float, default=1.0e5)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--px", type=int, default=160)
    parser.add_argument("--sensor-side", type=int, default=11)
    parser.add_argument("--step-mm", type=float, default=0.1)
    parser.add_argument("--tag", default="sensor_script")
    parser.add_argument("--out")
    parser.add_argument("--no-noise", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.object_size <= 0.0:
        raise SystemExit("--object-size must be positive")
    if args.drive == "manual" and args.runtime != "isaac":
        raise SystemExit("manual drive requires --runtime isaac")
    if args.runtime == "host":
        if args.drive != "script":
            raise SystemExit("host runtime only supports --drive script")
        report = run_script(args)
    else:
        report = run_interactive(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
