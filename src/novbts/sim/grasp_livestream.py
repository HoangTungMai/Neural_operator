"""Livestream helpers for the grasp demo: viewport framing and post-run hold.

Everything Isaac/omni is imported lazily inside the functions — these are only
callable after AppLauncher has started the SimulationApp.

Black-screen context (native livestream, mode 1): the isaaclab `rendering.kit`
experience ships with `exts."omni.renderer.core".present.enabled=false`, which
keeps NVENC at 0% (client connects on port 48010 but receives no video). The
driver re-enables the present thread via a `--/exts/omni.renderer.core/present/
enabled=true` kit override before AppLauncher starts (see grasp_demo).
"""
from __future__ import annotations

import time

import numpy as np

from novbts.sim.grasp_output import flog


def frame_viewport_on_robot(robot_prim_path: str = "/World/Robot") -> tuple[tuple[float, float, float] | None, float]:
    """Aim the default kit viewport camera (the one native streaming encodes) at the robot.

    The default "Perspective" camera looks at the origin, which clips the arm out of
    frame. Measure the robot's world bbox and auto-frame a pulled-back 3/4 view, so
    the view is right regardless of --robot-z / mode / arm pose. Also logs a one-shot
    STAGE_PRIM dump of /World children (visibility, center, size) for debugging.
    """
    robot_center = None
    robot_diag = 0.0
    try:
        import omni.usd as _omni_usd
        from pxr import Usd as _Usd, UsdGeom as _UsdGeom
        _stage = _omni_usd.get_context().get_stage()
        _bb = _UsdGeom.BBoxCache(_Usd.TimeCode.Default(),
                                 [_UsdGeom.Tokens.default_, _UsdGeom.Tokens.render])
        for _prim in _stage.GetPrimAtPath("/World").GetChildren():
            _p = _prim.GetPath().pathString
            try:
                _vis = str(_UsdGeom.Imageable(_prim).ComputeVisibility())
            except Exception:
                _vis = "?"
            try:
                _r = _bb.ComputeWorldBound(_prim).ComputeAlignedRange()
                _c = _r.GetMidpoint(); _s = _r.GetSize()
                flog(f"STAGE_PRIM {_p} vis={_vis} "
                     f"center=({_c[0]:.3f},{_c[1]:.3f},{_c[2]:.3f}) "
                     f"size=({_s[0]:.3f},{_s[1]:.3f},{_s[2]:.3f})")
                if _p == robot_prim_path and _vis != "invisible":
                    robot_center = (float(_c[0]), float(_c[1]), float(_c[2]))
                    robot_diag = float(np.linalg.norm([_s[0], _s[1], _s[2]]))
            except Exception as _e:
                flog(f"STAGE_PRIM {_p} vis={_vis} bbox_err={_e}")
    except Exception as exc:
        flog(f"STAGE_DUMP_FAIL {type(exc).__name__}: {exc}")

    try:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except Exception:
            from omni.isaac.core.utils.viewports import set_camera_view
        if robot_center is not None and robot_diag > 1e-6:
            # 3/4 view, pulled back proportional to the robot's bbox diagonal.
            _d = max(0.8, 2.2 * robot_diag)
            _dir = np.asarray([1.0, -0.85, 0.55], dtype=float)
            _dir /= np.linalg.norm(_dir)
            _eye = np.asarray(robot_center, dtype=float) + _d * _dir
            _tgt = list(robot_center)
        else:
            _eye = np.asarray([2.2, -1.8, 1.35])
            _tgt = [0.25, 0.0, 0.25]
        set_camera_view(eye=[float(v) for v in _eye], target=[float(v) for v in _tgt])
        flog(f"LIVESTREAM_VIEWPORT_CAMERA_SET eye=({_eye[0]:.3f},{_eye[1]:.3f},{_eye[2]:.3f}) "
             f"target=({_tgt[0]:.3f},{_tgt[1]:.3f},{_tgt[2]:.3f}) robot_diag={robot_diag:.3f}")
    except Exception as exc:
        flog(f"LIVESTREAM_VIEWPORT_CAMERA_FAIL {type(exc).__name__}: {exc}")
    return robot_center, robot_diag


def hold_livestream(args, app) -> dict:
    """Keep the SimulationApp updating after the run so a remote client can watch.

    `app` is the live SimulationApp (or None if Isaac never came up).
    """
    hold_s = float(getattr(args, "livestream_hold", 0.0))
    livestream_value = int(getattr(args, "livestream", 0))
    if livestream_value <= 0:
        return {"enabled": False, "reason": "livestream disabled"}
    if hold_s <= 0.0:
        return {"enabled": False, "reason": "hold <= 0"}
    info = {"enabled": True, "mode": livestream_value, "requested_s": hold_s, "updates": 0}
    if app is None:
        info.update({"completed": False, "error": "SimulationApp unavailable"})
        flog("LIVESTREAM_HOLD_FAIL SimulationApp unavailable")
        return info
    deadline = time.time() + hold_s
    flog(f"LIVESTREAM_HOLD_BEGIN seconds={hold_s:.3f}")
    try:
        while time.time() < deadline:
            if hasattr(app, "is_running") and not app.is_running():
                info["stopped_early"] = True
                break
            if hasattr(app, "update"):
                app.update()
                info["updates"] += 1
            time.sleep(1.0 / 30.0)
        info["completed"] = not info.get("stopped_early", False)
        flog(f"LIVESTREAM_HOLD_END updates={info['updates']} completed={info['completed']}")
    except Exception as exc:
        info.update({"completed": False, "error": f"{type(exc).__name__}: {exc}"})
        flog(f"LIVESTREAM_HOLD_FAIL {type(exc).__name__}: {exc}")
    return info
