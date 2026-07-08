#!/usr/bin/env python3
"""Container-only UR3e + Robotiq + rigid VBTS gelpad USD builder.

Run through ``infra/run_sim_grasp.sh``. This file is intentionally conservative:
P0 ``--check-only`` discovers real link/joint names first; the build path then
uses those names from ``asset_check.json`` instead of guessing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Container-only exception allowed by the task/GT-driver convention.
if "/work/src" not in sys.path:
    sys.path.append("/work/src")
for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


UR3E_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
ROBOTIQ_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
PROGRESS = "/work/sim_grasp_asset_progress.txt"
CHECK_JSON = "/work/data/assets/asset_check.json"
OUT_USD = "/work/data/assets/ur3e_robotiq_vbts.usd"
OUT_MANIFEST = "/work/data/assets/ur3e_robotiq_vbts.json"


def flog(msg: str) -> None:
    Path(PROGRESS).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def _url_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return 200 <= int(resp.status) < 300
    except Exception as exc:
        flog(f"HEAD failed for {url}: {type(exc).__name__}: {exc}")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return 200 <= int(resp.status) < 300
        except Exception as exc2:
            flog(f"GET failed for {url}: {type(exc2).__name__}: {exc2}")
            return False


def _import_usd():
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    _simulation_app = app_launcher.app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _open_stage(Usd, url: str):
    stage = Usd.Stage.Open(url)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open failed: {url}")
    return stage


def _prim_paths(stage):
    return [str(p.GetPath()) for p in stage.Traverse()]


def _joint_paths(stage):
    out = []
    for p in stage.Traverse():
        ty = p.GetTypeName()
        if "Joint" in ty:
            row = {"path": str(p.GetPath()), "type": ty}
            try:
                rel0 = p.GetRelationship("physics:body0").GetTargets()
                rel1 = p.GetRelationship("physics:body1").GetTargets()
                row["body0"] = [str(x) for x in rel0]
                row["body1"] = [str(x) for x in rel1]
            except Exception:
                pass
            out.append(row)
    return out


def _articulation_roots(UsdPhysics, stage):
    return [
        str(p.GetPath()) for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]


def _collision_enabled(UsdPhysics, prim) -> bool:
    attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
    val = attr.Get() if attr else None
    return True if val is None else bool(val)


def _collision_prims_under(UsdPhysics, stage, roots: list[str]) -> list[dict]:
    rows = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not any(path == root or path.startswith(root + "/") for root in roots):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        rows.append({
            "path": path,
            "type": prim.GetTypeName(),
            "enabled": _collision_enabled(UsdPhysics, prim),
        })
    return rows


def _set_collision_enabled(UsdPhysics, prim, enabled: bool) -> None:
    api = UsdPhysics.CollisionAPI.Apply(prim)
    attr = api.GetCollisionEnabledAttr()
    if not attr:
        attr = api.CreateCollisionEnabledAttr()
    attr.Set(bool(enabled))


def _pick_flange(paths: list[str]) -> str:
    candidates = [
        p for p in paths
        if any(k in p.lower() for k in ("tool0", "flange", "wrist_3"))
        and not any(k in p.lower() for k in ("collision", "visual", "mesh"))
    ]
    if not candidates:
        candidates = [p for p in paths if any(k in p.lower() for k in ("tool0", "flange", "wrist_3"))]
    if not candidates:
        raise RuntimeError("could not find UR flange/tool prim in UR3e asset")
    return sorted(candidates, key=lambda p: (("tool0" not in p.lower()), len(p)))[0]


def _pick_pad_links(paths: list[str]) -> list[str]:
    low = [(p, p.lower()) for p in paths]
    out = []
    for side in ("left", "right"):
        exact = [
            p for p, q in low
            if f"{side}_inner_finger" in q
            and "joint" not in q
            and "defeatured" not in q
            and "pad_open" not in q
        ]
        if exact:
            out.append(sorted(exact, key=len)[0])
            continue
        mesh = [p for p, q in low if side in q and "inner_finger" in q and "fingertip" in q]
        if mesh:
            out.append(sorted(mesh, key=len)[0])
    if len(out) < 2:
        pads = [p for p, q in low if "finger" in q and ("left" in q or "right" in q)]
        out.extend([p for p in sorted(pads) if p not in out])
    return out[:2]


def _map_ur_path(path: str) -> str:
    return "/Robot" + path[len("/ur3e"):] if path.startswith("/ur3e") else path


def _map_robotiq_path(path: str) -> str:
    prefix = "/World/Robotiq_2F_85"
    return "/Robot/robotiq_2f85/Robotiq_2F_85" + path[len(prefix):] if path.startswith(prefix) else path


def _remove_articulation_roots_under(UsdPhysics, stage, prefix: str) -> list[str]:
    removed = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prefix) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed.append(path)
    return removed


def _quat_from_R(Gf, R):
    """Return Gf.Quatf from a 3x3 rotation matrix with columns as local axes."""
    import math

    m = R
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _dot(a, b) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _cross(a, b) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _sub(a, b) -> tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _add(a, b) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _scale(a, s: float) -> tuple[float, float, float]:
    return (float(a[0]) * float(s), float(a[1]) * float(s), float(a[2]) * float(s))


def _norm(a) -> float:
    import math

    return math.sqrt(max(_dot(a, a), 0.0))


def _normalize(a) -> tuple[float, float, float]:
    n = _norm(a)
    if n <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector: {a}")
    return (float(a[0]) / n, float(a[1]) / n, float(a[2]) / n)


def _frame_from_prim(Gf, Usd, UsdGeom, stage, path: str) -> dict:
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"prim not found: {path}")
    mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    origin = mat.ExtractTranslation()
    axes = []
    for vec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        axis = mat.TransformDir(Gf.Vec3d(*vec))
        axes.append(_normalize((float(axis[0]), float(axis[1]), float(axis[2]))))
    return {
        "path": path,
        "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
        "axes": tuple(axes),
    }


def _transform_point_from_frame(frame: dict, point) -> tuple[float, float, float]:
    out = frame["origin"]
    for value, axis in zip(point, frame["axes"]):
        out = _add(out, _scale(axis, float(value)))
    return out


def _signed_local_axis_for_direction(frame: dict, target_w) -> tuple[tuple[float, float, float], float]:
    target = _normalize(target_w)
    candidates = (
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    )
    best = None
    for local in candidates:
        world = (
            local[0] * frame["axes"][0][0] + local[1] * frame["axes"][1][0] + local[2] * frame["axes"][2][0],
            local[0] * frame["axes"][0][1] + local[1] * frame["axes"][1][1] + local[2] * frame["axes"][2][1],
            local[0] * frame["axes"][0][2] + local[1] * frame["axes"][1][2] + local[2] * frame["axes"][2][2],
        )
        score = _dot(_normalize(world), target)
        if best is None or score > best[1]:
            best = (local, score)
    if best is None or best[1] < 0.70:
        raise RuntimeError(f"could not resolve inward local pad axis for target {target_w}; best={best}")
    return best


def _gel_R_from_local_normal(local_normal) -> tuple[tuple[float, float, float], ...]:
    z = _normalize(local_normal)
    hint = (1.0, 0.0, 0.0)
    if abs(_dot(hint, z)) > 0.90:
        hint = (0.0, 1.0, 0.0)
    x = _normalize(_sub(hint, _scale(z, _dot(hint, z))))
    y = _normalize(_cross(z, x))
    return (
        (x[0], y[0], z[0]),
        (x[1], y[1], z[1]),
        (x[2], y[2], z[2]),
    )


def _derive_gel_authoring(Gf, Usd, UsdGeom, stage, pad_specs: list[dict]) -> dict:
    if len(pad_specs) != 2:
        raise RuntimeError(f"expected exactly two pad specs, got {len(pad_specs)}")
    frames = {
        spec["side"]: _frame_from_prim(Gf, Usd, UsdGeom, stage, spec["pad_link"])
        for spec in pad_specs
    }
    anchors_w = {
        spec["side"]: _transform_point_from_frame(frames[spec["side"]], spec["anchor_link"])
        for spec in pad_specs
    }
    gap_vec = _sub(anchors_w["right"], anchors_w["left"])
    if _norm(gap_vec) <= 1.0e-6:
        raise RuntimeError(f"gel anchor points are coincident at rest: {anchors_w}")
    authoring = {}
    for spec in pad_specs:
        side = spec["side"]
        other = "right" if side == "left" else "left"
        target_w = _sub(anchors_w[other], anchors_w[side])
        local_normal, axis_score = _signed_local_axis_for_direction(frames[side], target_w)
        authoring[side] = {
            "translation": tuple(float(x) for x in spec["anchor_link"]),
            "R": _gel_R_from_local_normal(local_normal),
            "inward_local_axis": list(local_normal),
            "inward_axis_score": float(axis_score),
            "anchor_w": list(anchors_w[side]),
        }
    return authoring


def _gel_stage_geometry_assertion(Gf, Usd, UsdGeom, stage, pads: list[dict]) -> dict:
    frames = {}
    for pad in pads:
        frame = _frame_from_prim(Gf, Usd, UsdGeom, stage, pad["gel_frame_prim"])
        frames[pad["name"]] = {
            "origin": list(frame["origin"]),
            "x": list(frame["axes"][0]),
            "y": list(frame["axes"][1]),
            "normal": list(frame["axes"][2]),
        }
    if "left" not in frames or "right" not in frames:
        raise RuntimeError(f"expected left/right gel frames, got {sorted(frames)}")
    left = frames["left"]
    right = frames["right"]
    left_n = _normalize(left["normal"])
    right_n = _normalize(right["normal"])
    delta_lr = _sub(right["origin"], left["origin"])
    normal_dot = _dot(left_n, right_n)
    left_face_gap = _dot(delta_lr, left_n)
    right_face_gap = _dot(_scale(delta_lr, -1.0), right_n)
    face_gap = 0.5 * (left_face_gap + right_face_gap)
    origin_gap = _norm(delta_lr)
    metrics = {
        "frames": frames,
        "normal_dot": float(normal_dot),
        "face_gap_m": float(face_gap),
        "left_face_gap_m": float(left_face_gap),
        "right_face_gap_m": float(right_face_gap),
        "origin_gap_m": float(origin_gap),
    }
    if normal_dot >= -0.95:
        raise RuntimeError(f"gel normals are not opposing at rest: {metrics}")
    if not (0.005 <= face_gap <= 0.100):
        raise RuntimeError(f"gel face gap is not plausible at rest: {metrics}")
    return metrics


def _pick_gripper_joint(joints: list[dict]) -> str:
    for key in ("finger_joint", "knuckle", "driver"):
        vals = [j["path"] for j in joints if key in j["path"].lower()]
        if vals:
            return sorted(vals)[0]
    return joints[0]["path"] if joints else ""


def check_only() -> dict:
    Path("/work/data/assets").mkdir(parents=True, exist_ok=True)
    Path(PROGRESS).write_text("")
    flog("CHECK start")
    if not _url_ok(UR3E_URL) or not _url_ok(ROBOTIQ_URL):
        raise SystemExit("asset URL check failed")
    flog("URL checks OK")
    _Gf, _Sdf, Usd, _UsdGeom, UsdPhysics, _UsdShade = _import_usd()
    ur_stage = _open_stage(Usd, UR3E_URL)
    gr_stage = _open_stage(Usd, ROBOTIQ_URL)
    ur_paths = _prim_paths(ur_stage)
    gr_paths = _prim_paths(gr_stage)
    gr_joints = _joint_paths(gr_stage)
    pad_links = _pick_pad_links(gr_paths)
    data = {
        "ur3e_url": UR3E_URL,
        "robotiq_url": ROBOTIQ_URL,
        "ur3e_prim_count": len(ur_paths),
        "robotiq_prim_count": len(gr_paths),
        "ur3e_articulation_roots": _articulation_roots(UsdPhysics, ur_stage),
        "robotiq_articulation_roots": _articulation_roots(UsdPhysics, gr_stage),
        "ur3e_flange_link": _pick_flange(ur_paths),
        "robotiq_joints": gr_joints,
        "robotiq_links": [p for p in gr_paths if "joint" not in p.lower()],
        "robotiq_pad_links_guess": pad_links,
        "robotiq_inner_finger_collision_prims": _collision_prims_under(UsdPhysics, gr_stage, pad_links),
        "robotiq_drive_joint_guess": _pick_gripper_joint(gr_joints),
    }
    Path(CHECK_JSON).write_text(json.dumps(data, indent=2))
    flog(f"CHECK_OK wrote {CHECK_JSON}")
    print("ASSET_CHECK_OK", CHECK_JSON, flush=True)
    return data


def _define_cube(UsdGeom, UsdPhysics, UsdShade, material, stage, path: str, *, translate, scale, collision: bool, visible: bool):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(translate)
    xform.AddScaleOp().Set(scale)
    if not visible:
        prim.CreateVisibilityAttr("invisible")
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        from pxr import PhysxSchema

        physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
        physx_col.CreateContactOffsetAttr(2.0e-4)
        physx_col.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    return prim


def _make_physics_material(UsdPhysics, UsdShade, stage, path: str, mu: float):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    return mat


def build(gripper_collision_mode: str = "default") -> dict:
    Path(PROGRESS).write_text("")
    flog("BUILD start")
    Path("/work/data/assets").mkdir(parents=True, exist_ok=True)
    if not Path(CHECK_JSON).exists():
        flog("asset_check missing; running check_only first")
        check_only()
    info = json.loads(Path(CHECK_JSON).read_text())
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _import_usd()
    stage = Usd.Stage.CreateNew(OUT_USD + ".tmp.usda")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim("/Robot", "Xform")
    stage.SetDefaultPrim(robot)
    robot.GetReferences().AddReference(UR3E_URL)
    gripper = stage.DefinePrim("/Robot/robotiq_2f85", "Xform")
    gripper.GetReferences().AddReference(ROBOTIQ_URL)
    removed_roots = _remove_articulation_roots_under(UsdPhysics, stage, "/Robot/robotiq_2f85")
    flog(f"removed gripper articulation roots: {removed_roots}")

    # Best-effort weld: exact bodies are verified after flatten/spawn.
    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/flange_to_robotiq_fixed")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(_map_ur_path(info["ur3e_flange_link"]))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/robotiq_2f85/Robotiq_2F_85/base_link")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    gel_mat = _make_physics_material(UsdPhysics, UsdShade, stage, "/Robot/vbts_gel_physics_material", 0.9)

    pad_links = info.get("robotiq_pad_links_guess") or ["/Robot/robotiq_2f85"]
    while len(pad_links) < 2:
        pad_links.append(pad_links[-1])
    mapped_pad_links = [_map_robotiq_path(p) for p in pad_links[:2]]
    robotiq_collision_prims = _collision_prims_under(UsdPhysics, stage, ["/Robot/robotiq_2f85"])
    original_inner_finger_collisions = _collision_prims_under(UsdPhysics, stage, mapped_pad_links)
    disabled_original_collisions = []
    for row in original_inner_finger_collisions:
        prim = stage.OverridePrim(row["path"])
        _set_collision_enabled(UsdPhysics, prim, False)
        disabled_original_collisions.append(row["path"])
    flog(f"disabled original inner-finger collisions: {disabled_original_collisions}")
    gel_anchor_link = {
        "left": (0.0243, -0.0319, 0.0),
        "right": (0.0258, -0.0155, 0.0),
    }
    pad_specs = [
        {
            "side": "left" if idx == 0 else "right",
            "pad_link": _map_robotiq_path(link),
            "anchor_link": gel_anchor_link["left" if idx == 0 else "right"],
        }
        for idx, link in enumerate(pad_links[:2])
    ]
    gel_authoring = _derive_gel_authoring(Gf, Usd, UsdGeom, stage, pad_specs)
    manifest_pads = []
    for idx, link in enumerate(pad_links[:2]):
        side = "left" if idx == 0 else "right"
        parent = stage.OverridePrim(_map_robotiq_path(link))
        base_path = str(parent.GetPath()) + f"/vbts_{side}"
        t = gel_authoring[side]["translation"]
        R = gel_authoring[side]["R"]
        gf = UsdGeom.Xform.Define(stage, base_path + "/gel_frame")
        gxf = UsdGeom.Xformable(gf)
        gxf.AddTranslateOp().Set(Gf.Vec3f(*t))
        gxf.AddOrientOp().Set(_quat_from_R(Gf, R))
        _define_cube(UsdGeom, UsdPhysics, UsdShade, gel_mat, stage, base_path + "/gel_frame/housing",
                     translate=Gf.Vec3f(0.0, 0.0, -0.005),
                     scale=Gf.Vec3f(0.011, 0.008, 0.004),
                     collision=True, visible=True)
        _define_cube(UsdGeom, UsdPhysics, UsdShade, None, stage, base_path + "/gel_frame/gel_visual",
                     translate=Gf.Vec3f(0.0, 0.0, -0.0015),
                     scale=Gf.Vec3f(0.010, 0.010, 0.0015),
                     collision=False, visible=True)
        _define_cube(UsdGeom, UsdPhysics, UsdShade, gel_mat, stage, base_path + "/gel_frame/gel_collision",
                     translate=Gf.Vec3f(0.0, 0.0, -0.00165),
                     scale=Gf.Vec3f(0.010, 0.010, 0.00135),
                     collision=True, visible=False)
        manifest_pads.append({
            "name": side,
            "pad_link": str(parent.GetPath()),
            "gel_frame_prim": base_path + "/gel_frame",
            "gel_collision_prim": base_path + "/gel_frame/gel_collision",
            "inward_local_axis": gel_authoring[side]["inward_local_axis"],
            "inward_axis_score": gel_authoring[side]["inward_axis_score"],
            "T_link_gel": [
                [R[0][0], R[0][1], R[0][2], t[0]],
                [R[1][0], R[1][1], R[1][2], t[1]],
                [R[2][0], R[2][1], R[2][2], t[2]],
                [0, 0, 0, 1.0],
            ],
        })

    rest_gel_geometry = _gel_stage_geometry_assertion(Gf, Usd, UsdGeom, stage, manifest_pads)

    disabled_gripper_collision_prims = []
    if gripper_collision_mode == "gel-only":
        keep = {p["gel_collision_prim"] for p in manifest_pads}
        for row in _collision_prims_under(UsdPhysics, stage, ["/Robot/robotiq_2f85"]):
            if row["path"] in keep:
                continue
            prim = stage.OverridePrim(row["path"])
            _set_collision_enabled(UsdPhysics, prim, False)
            disabled_gripper_collision_prims.append(row["path"])
        flog(f"gel-only gripper collision mode disabled: {disabled_gripper_collision_prims}")

    flat = stage.Flatten()
    flat.Export(OUT_USD)
    manifest = {
        "usd_path": OUT_USD,
        "source_asset_check": CHECK_JSON,
        "ur3e_flange_link": _map_ur_path(info["ur3e_flange_link"]),
        "arm_joints_expr": ["shoulder.*", "elbow.*", "wrist.*"],
        "gripper_drive_joint": _map_robotiq_path(info.get("robotiq_drive_joint_guess", "finger_joint")),
        "pads": manifest_pads,
        "rest_gel_geometry": rest_gel_geometry,
        "gel": {
            "xy_m": 0.020,
            "z_m": 0.003,
            "collision_z_m": 0.0027,
            "recess_m": 0.0003,
            "mu": 0.9,
            "contact_offset_m": 2.0e-4,
            "rest_offset_m": 0.0,
            "housing_collision_enabled": gripper_collision_mode != "gel-only",
        },
        "gripper_collision_mode": gripper_collision_mode,
        "removed_articulation_roots": removed_roots,
        "robotiq_collision_prims_before_vbts": robotiq_collision_prims,
        "original_inner_finger_collision_prims": original_inner_finger_collisions,
        "disabled_original_inner_finger_collision_prims": disabled_original_collisions,
        "disabled_gripper_collision_prims": disabled_gripper_collision_prims,
    }
    Path(OUT_MANIFEST).write_text(json.dumps(manifest, indent=2))
    flog(f"ASSET_BUILD_OK wrote {OUT_USD} and {OUT_MANIFEST}")
    print("ASSET_BUILD_OK", OUT_USD, flush=True)
    return manifest


def verify() -> dict:
    Gf, _Sdf, Usd, UsdGeom, UsdPhysics, _UsdShade = _import_usd()
    if not Path(OUT_USD).exists() or not Path(OUT_MANIFEST).exists():
        raise SystemExit("USD/manifest missing; run build first")
    stage = Usd.Stage.Open(OUT_USD)
    roots = _articulation_roots(UsdPhysics, stage)
    manifest = json.loads(Path(OUT_MANIFEST).read_text())
    gel_missing = [
        p["gel_collision_prim"] for p in manifest["pads"]
        if stage.GetPrimAtPath(p["gel_collision_prim"]) is None
        or not stage.GetPrimAtPath(p["gel_collision_prim"]).IsValid()
    ]
    from novbts.sim.vbts_sensor import VBTSSensor

    rest_gel_geometry = _gel_stage_geometry_assertion(Gf, Usd, UsdGeom, stage, manifest["pads"])

    sensor = VBTSSensor("/work/data/assets/vbts_fno/lr_fno_realistic_bc.pt", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None)
    import numpy as np
    out = sensor.render(np.zeros((sensor.side, sensor.side), dtype=np.float32), 0.0, 0.0, noise=False)
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=OUT_USD),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=80.0,
                damping=8.0,
                effort_limit=5.0,
            )
        },
    ))
    sim.reset()
    robot.update(1.0 / 120.0)
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    joints = _joint_paths(stage)
    inner_finger_collision_prims = _collision_prims_under(
        UsdPhysics, stage, [p["pad_link"] for p in manifest.get("pads", [])]
    )
    robotiq_collision_prims = _collision_prims_under(UsdPhysics, stage, ["/Robot/robotiq_2f85"])
    enabled_robotiq_collision_prims = [row for row in robotiq_collision_prims if row["enabled"]]
    data = {
        "usd": OUT_USD,
        "manifest": OUT_MANIFEST,
        "articulation_roots": roots,
        "runtime_joint_names": joint_names,
        "runtime_body_names": body_names,
        "joint_paths": joints,
        "inner_finger_collision_prims": inner_finger_collision_prims,
        "robotiq_collision_prims": robotiq_collision_prims,
        "enabled_robotiq_collision_prims": enabled_robotiq_collision_prims,
        "gel_missing": gel_missing,
        "rest_gel_geometry": rest_gel_geometry,
        "fno_rest_img_shape": list(out["img"].shape),
    }
    if gel_missing:
        raise SystemExit(f"gel prims missing: {gel_missing}")
    if len(roots) != 1:
        raise SystemExit(f"expected exactly 1 ArticulationRootAPI, found {len(roots)}: {roots}")
    if not any("finger" in name.lower() or "knuckle" in name.lower() for name in joint_names):
        raise SystemExit(f"gripper joints missing from runtime articulation: {joint_names}")
    Path("/work/data/assets/asset_verify.json").write_text(json.dumps(data, indent=2))
    flog("ASSET_VERIFY_OK")
    print("ASSET_VERIFY_OK", flush=True)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gripper-collision-mode", choices=["default", "gel-only"], default="default")
    args = ap.parse_args()
    if args.check_only:
        check_only()
    elif args.verify:
        verify()
    else:
        build(args.gripper_collision_mode)
    os._exit(0)


if __name__ == "__main__":
    main()
