#!/usr/bin/env python3
"""
Mesh-resolution probe for the deformable gel — find the finest hexahedral
resolution that still COOKS in reasonable time (high res hangs at init).

Logs live to /work/fem_progress.txt (Isaac swallows stdout) so a watchdog can
catch a cooking hang.  Reports: total nodes, top-surface nodes, and estimated
nodes across the R=0.02/d=0.005 contact diameter (target >=8-10 for stick core).

Run ONE config per container:
  ... isaac_mesh_probe.py --hex-res 20 --gel-xy 0.05
"""
import argparse, sys, glob, time, traceback

for _p in glob.glob("/workspace/isaaclab/source/*"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ap = argparse.ArgumentParser()
ap.add_argument("--hex-res", type=int, default=-1, help="-1 = leave default")
ap.add_argument("--gel-xy", type=float, default=0.10)
ap.add_argument("--gel-z", type=float, default=0.04)
args = ap.parse_args()

_PROG = "/work/fem_progress.txt"
def flog(m):
    with open(_PROG, "a") as f:
        f.write(m + "\n"); f.flush()
open(_PROG, "w").close()
flog(f"probe start hex_res={args.hex_res} gel_xy={args.gel_xy}")

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True).app
flog("app created")

import numpy as np, torch, dataclasses
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import DeformableObject, DeformableObjectCfg

try:
    # introspect the resolution field name + default
    fields = {f.name: f.default for f in dataclasses.fields(sim_utils.DeformableBodyPropertiesCfg)}
    flog("DeformableBodyPropertiesCfg fields: " + ", ".join(fields.keys()))
    res_field = next((k for k in fields if "hex" in k.lower() or "resolution" in k.lower()), None)
    flog(f"resolution field = {res_field} (default={fields.get(res_field)})")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    dp_kwargs = dict(rest_offset=0.0, contact_offset=0.002, solver_position_iteration_count=30)
    if args.hex_res > 0 and res_field is not None:
        dp_kwargs[res_field] = args.hex_res
    flog(f"deformable_props kwargs: {dp_kwargs}")

    cfg = DeformableObjectCfg(
        prim_path="/World/gel",
        spawn=sim_utils.MeshCuboidCfg(
            size=(args.gel_xy, args.gel_xy, args.gel_z),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(**dp_kwargs),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=1.0e5, poissons_ratio=0.45, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.gel_z / 2.0)),
    )
    flog("cfg built; creating DeformableObject (cooking starts)")
    t0 = time.perf_counter()
    gel = DeformableObject(cfg)
    flog(f"DeformableObject created t={time.perf_counter()-t0:.1f}s; sim.reset()")
    sim.reset()
    cook = time.perf_counter() - t0
    flog(f"sim.reset OK; total cook/init={cook:.1f}s")

    rest = gel.data.nodal_pos_w[0].cpu().numpy()
    zmax = rest[:, 2].max()
    top = np.where(rest[:, 2] > zmax - 2e-3)[0]
    # nodes across contact diameter: top-node spacing in x near center
    top_xy = rest[top, :2]
    a = np.sqrt(0.02 * 0.005)  # Hertz contact radius (m)
    # median nearest-neighbour spacing among top nodes
    from scipy.spatial import cKDTree
    d2, _ = cKDTree(top_xy).query(top_xy, k=2)
    spacing = float(np.median(d2[:, 1]))
    flog(f"RESULT nodes_total={rest.shape[0]} top_nodes={top.shape[0]} "
         f"top_spacing={spacing*1000:.2f}mm contact_diam={2*a*1000:.1f}mm "
         f"nodes_across_contact={2*a/spacing:.1f} cook_s={cook:.1f}")
    print(f"MESH_PROBE_OK nodes={rest.shape[0]} top={top.shape[0]} "
          f"across_contact={2*a/spacing:.1f} cook={cook:.1f}s")
except Exception:
    flog("MESH_PROBE_FAIL\n" + traceback.format_exc())
    print("MESH_PROBE_FAIL")

try:
    simulation_app.close()
except Exception:
    pass
flog("probe done")
