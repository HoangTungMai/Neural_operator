# Phase 5 — Marker-dot VBTS sensor model (differentiable camera + dot rendering)

> First execution step: copy this plan to `docs/phase5_plan.md` in the repo so it lives in the workspace.

## Context

The project's end goal is **both** tracks: (A) a complete Isaac Sim VBTS DIY environment and
(B) the differentiable FNO surrogate. Track B is mature — Gate 3 closed and Phase 4 proved the
differentiable surrogate trains a control policy ~64× more sample-efficiently than gradient-free
ES (and the Suh contact-gradient pathology doesn't bite because the FNO smooths the stick→slip
boundary). Track A is only ~30%: `isaac_extract_shear.py` produces the gel-surface **marker
displacement field** `disp[N,M,3]`, but there is **no actual sensor output** — no camera, no dot
image, no marker flow. A real marker-dot VBTS (GelSight-DIY / dotted-gel + camera) outputs a
**camera image of dots**, from which 2D marker flow is tracked.

Target sensor (user-chosen): **marker-dot + camera** — the lightest-optics path. Crucial enabler
found in exploration: the generator already gives the surface displacement field, and deformed dot
positions are simply `coords + disp[:, :2]` at height `z_rest + disp[:, 2]`. So the sensor image is
a **projection + rendering of an existing field** (no new physics solve). Built in torch it is a
**differentiable renderer** → `FNO(contact) → disp → sensor image` is end-to-end differentiable,
composing with Phase 4 (inverse/control now from the actual marker image, not just `disp`).

**Phase 5 committed scope = the sensor model (milestones 5a + 5b).** It delivers the defining
missing piece, reuses existing FEM GT, and carries **zero Isaac/Docker/deadlock risk**. The
heavier, riskier environment pieces (5c deeper-contact + object meshes in Isaac Sim; 5d DIY
calibration; 5e Isaac-Lab RL-env wrapper) are sequenced as the explicit follow-on roadmap below,
not built in this phase.

## Milestone 5a — Differentiable marker-dot sensor model (PRIMARY)

New subpackage `src/novbts/sensor/` (with `__init__.py`):

### `src/novbts/sensor/markercam.py` (torch, differentiable)
```python
class PinholeCamera:                 # K from (fov_deg, px_w, px_h, working_dist); pose looks at gel membrane
    def project(self, pts_xyz):      # [N,M,3] gel-frame -> [N,M,2] pixel coords (perspective divide)

def deformed_marker_xyz(coords, disp, z_rest):   # rest grid + disp -> [N,M,3] deformed dot positions (gel frame)
def render_dots(pix, px_h, px_w, dot_sigma):     # [N,M,2] -> [N,1,px_h,px_w] image; torch Gaussian splat (differentiable)
def track_flow_known(pix_rest, pix_def):         # [N,M,2] pixel flow = pix_def - pix_rest (correspondence known by construction)
def track_flow_image(img_rest, img_def):         # optional: blob-centroid re-detection (round-trip validation only)
```
- **Camera convention (default, documented & overridable):** pinhole *below* the membrane looking
  up the gel +z axis (standard for dotted-gel VBTS), orthographic-ish at a working distance that
  maps the gel footprint to the pixel frame. Expose `--fov-deg`, `--px`, `--working-dist`.
- **Renderer is the sensor:** sits AFTER GT/FNO. The FNO is **unchanged** (still predicts `disp`);
  the sensor is a deterministic differentiable map `disp -> image`. Gaussian-splat render keeps it
  smooth/differentiable (no hard rasterization).

### `src/novbts/sensor/build_sensor_dataset.py` (CLI)
- Loads an aggregated FEM npz (default `data/fem/shear_fine_swept_normaug.npz`), reads
  `coords[M,2]`, `disp[N,M,3]`, `mode[N]` (reuse loader pattern from `fem_benchmark.load`).
- Produces the **sensor dataset**: `rest_img[1,1,H,W]`, `def_img[N,1,H,W]` (or store pixel
  positions + render lazily to save disk), `pix_flow[N,M,2]`, plus `K`, `pose`, and a back-link to
  the underlying `params`/`mode`. Write to `data/fem/<stem>_sensor.npz`.
- Diagnostics to `runs/phase5/`: a preview montage PNG (rest vs deformed dot image for a
  representative full-slip frame) + round-trip error table.

**Reuse:** `coords/disp/mode` straight from the npz (produced by `isaac_extract_shear.py`
marker_grid/sample_to_markers); `paths.FEM/RUNS/ensure`; matplotlib `Agg` plotting pattern from
`field2field.py` main; printout/JSON conventions from `inverse_demo.py`.

## Milestone 5b — FNO compatibility + end-to-end differentiable sensor pipeline

Demonstrate the sensor composes with the existing surrogate and Phase 4 machinery:
- **Compatibility check:** `FEM disp -> render` vs `FNO(contact) -> disp -> render` produce
  consistent marker images (rel-L2 in pixel-flow space), confirming the FNO + renderer reproduces
  the sensor observation. FNO input format `params_to_fieldinput` (`field2field.py`) is untouched.
- **Differentiable round-trip demo** (`src/novbts/sensor/sensor_inverse_demo.py`, mirrors
  `inverse_demo.py`): recover applied shear `(sx,sy)` from the **rendered marker image** by
  autograd through `render ∘ FNO` — i.e. inverse problem from the *actual sensor output*, not the
  raw field. Report recovery error + that gradients flow through the renderer (the integrated
  Track-A⇄Track-B payoff).
- Output `runs/phase5/sensor_compat.json` + `sensor_inverse.json`.

## Follow-on roadmap (NOT in this phase — sequenced next)
- **5c** Extend `isaac_extract_shear.py`: `--indentor-geom {sphere,flat,mesh}` + `--indentor-mesh`
  (MeshFileCfg) for diverse object geometry; staged-loading attempt for deeper contact (deadlock-
  risk — keep shallow as fallback, `--smoke` gated). Docker `isaac-lab-fem`, headless.
- **5d** DIY calibration: config mapping sim camera intrinsics/pose + gel/marker layout to a
  physical build; fit script when hardware exists (sim2real).
- **5e** Isaac-Lab / gym RL-env wrapper: obs = sensor marker-flow image, action = contact/robot
  action, fast transition via FNO surrogate, reward for a tactile task; plug in the Phase-4 policy.

## Critical files
- NEW `src/novbts/sensor/__init__.py`, `markercam.py`, `build_sensor_dataset.py`,
  `sensor_inverse_demo.py`
- REUSE (read-only): `src/novbts/operator/field2field.py` (FNOField, params_to_fieldinput,
  train_operator, DEV), `src/novbts/operator/fem_benchmark.py` (load, norm_from),
  `src/novbts/operator/inverse_demo.py` (autograd-recovery pattern to mirror),
  `src/novbts/paths.py` (FEM, RUNS, ensure)
- Data in: `data/fem/shear_fine_swept_normaug.npz`; data out: `data/fem/*_sensor.npz`;
  diagnostics: `runs/phase5/`

## Verification
1. `python -m novbts.sensor.build_sensor_dataset --data data/fem/shear_fine_swept_normaug.npz`
   → writes `*_sensor.npz` + `runs/phase5/preview.png`; round-trip error (image-tracked flow vs
   projected `disp[:, :2]`) below tolerance for stick and slip frames.
2. Differentiability sanity: finite-difference vs autograd of `render` w.r.t. a marker position
   agree (the renderer is smooth/differentiable).
3. `python -m novbts.sensor.sensor_inverse_demo` → recovers `(sx,sy)` from the rendered marker
   image through `render ∘ FNO` to low error (compare to `inverse_demo.py`'s 2.3% from raw field),
   proving the end-to-end differentiable sensor pipeline.
4. Visual check: `runs/phase5/preview.png` shows rest vs deformed dot pattern with sensible
   shear-direction dot motion on a full-slip frame.

## Housekeeping
Phase 4 is complete (re-run finished; `runs/phase4/policy_servo.json` + curve). Commit the Phase 4
batch (`diff_policy.py` policy harness) + a `phase4-status` memory before starting Phase 5 coding,
so Phase 5 begins on a clean tree on branch `phase4-diff-policy` (or a fresh `phase5-sensor` branch).
