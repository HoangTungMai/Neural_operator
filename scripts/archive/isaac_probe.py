#!/usr/bin/env python3
"""
Capability probe for Isaac Lab DeformableObject (PhysX FEM) — writes findings
to /work/probe_result.txt (Isaac swallows stdout, so we log to a mounted file).

Run:
  docker run --rm --gpus all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$PWD":/work --entrypoint /isaac-sim/python.sh isaac-lab-base:latest \
    /work/scripts/isaac_probe.py
"""
OUT = "/work/probe_result.txt"
_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    _lines.append(s)
    try:
        with open(OUT, "w") as f:
            f.write("\n".join(_lines) + "\n")
    except Exception:
        pass

import sys, glob, traceback
# isaaclab editable install isn't picked up in this image; add source dirs.
for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    log("AppLauncher OK")
except Exception:
    log("AppLauncher FAIL\n" + traceback.format_exc())
    raise

try:
    import isaaclab
    log("isaaclab version:", getattr(isaaclab, "__version__", "?"))
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import DeformableObject, DeformableObjectCfg
    log("imports OK (DeformableObject available)")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    # ground
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    cfg = DeformableObjectCfg(
        prim_path="/World/gel",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.04),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0, contact_offset=0.001,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=1.0e5, poissons_ratio=0.45,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )
    log("DeformableObjectCfg built")
    obj = DeformableObject(cfg)
    log("DeformableObject created")

    sim.reset()
    log("sim.reset OK")
    for i in range(10):
        sim.step()
        obj.update(sim.get_physics_dt())

    pos = obj.data.nodal_pos_w
    log("nodal_pos_w shape:", tuple(pos.shape), "dtype:", str(pos.dtype))
    log("nodal_pos_w sample[0,:3]:", pos[0, :3].tolist())
    # rest/kinematic targets API for prescribed motion?
    for attr in ["nodal_kinematic_target", "nodal_pos_w", "nodal_vel_w",
                 "default_nodal_state_w"]:
        log("  has obj.data." + attr + ":", hasattr(obj.data, attr))
    for attr in ["write_nodal_pos_to_sim", "write_nodal_kinematic_target_to_sim",
                 "write_nodal_state_to_sim"]:
        log("  has obj." + attr + ":", hasattr(obj, attr))
    log("DEFORMABLE_BUILD_OK")
except Exception:
    log("DEFORMABLE_BUILD_FAIL\n" + traceback.format_exc())

try:
    simulation_app.close()
except Exception:
    pass
log("PROBE_DONE")
