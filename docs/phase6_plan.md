# Phase 6 — Complete the VBTS sensor-simulation framework (4 directions)

## Context

Phase 5 delivered a differentiable marker-dot sensor model (camera + dot render) on top of the FNO
surrogate + FEM GT. To turn the pile of scripts into a *complete, usable sensor-simulation
framework* (Track A of the project's two-track goal), four gaps remain. The user asked for all four,
sequenced by value/risk. Docker is available with `isaac-lab-fem` AND `isaac-lab-tacex` (GIPC deep
contact), so the Isaac-dependent pieces are feasible but slow (FEM ~0.5–4.6 s/frame); the
pure-Python pieces run immediately.

Sequencing (low-risk/high-value first → riskiest last):
**6a env wrapper → 6b realism/sim2real → 6c temporal+loading-history → 6d object-geometry/deep-contact.**

## 6a — Differentiable tactile ENV wrapper (pure Python, runnable now) [FRAMEWORK CORE]
Package FNO + sensor + control into one usable API/env — this is what makes it a "framework".
- New `src/novbts/sensor/tactile_env.py`: a Gym-style single-step contextual env (honest: FNO is a
  static one-step map, so step() is contextual, not multi-step dynamics).
  - `reset()` → samples a context (object μ,E,R + target tactile state); returns observation =
    rendered marker image (+ optional flow).
  - `step(action)` where action = contact params (sx,sy[,depth]); transition via frozen FNO →
    `disp` → `render` → sensor image; reward = −‖obs − target‖ (servo) or task-specific; fully
    differentiable (exposes `.differentiable_step` returning torch tensors with grad).
  - A thin Gym `Env` adapter (obs/action spaces) if `gymnasium` import succeeds; else a plain class.
- Integrate the Phase-4 policy: a demo that runs the trained PolicyMLP in the env.
- Reuse: `FNOField`/`params_to_fieldinput` (field2field), `markercam.*`, `diff_policy.PolicyMLP`,
  the frozen-FNO setup pattern. Output: `runs/phase6/env_demo.json` + a rollout preview PNG.
- Verify: `python -m novbts.sensor.tactile_env --demo` runs reset/step, reward improves under the
  Phase-4 policy vs random action; differentiable_step passes a gradcheck (grad flows to action).

## 6b — Sensor realism + sim2real scaffold (pure Python, runnable now)
Make the rendered image believable and sim2real-ready.
- Extend `markercam.py` (or new `sensor/realism.py`): additive camera noise (Gaussian + optional
  Poisson), per-marker intensity/size jitter, optional blur; a REAL blob-detection tracker
  (local-max + weighted centroid / simple matching) distinct from the known-correspondence flow,
  to measure tracking error under noise.
- `sensor/calibration.py`: a `SensorCalib` config (camera intrinsics/pose, marker layout, gel
  scale, noise levels) serialisable to JSON; a `fit_to_captures()` stub that calibrates sim params
  to a few real frames (no-op until hardware — documented).
- Verify: round-trip tracking error vs noise level curve (`runs/phase6/noise_robustness.json` +
  PNG); blob tracker recovers flow within tolerance at low noise, degrades gracefully.

## 6c — Temporal sequences + loading history (cheap rollout now; true FEM via Docker)
VBTS is a *video* sensor and slip is path-dependent; the FEM generator already micro-steps the drag
but keeps only the final frame (isaac_extract_shear.py:269).
- **Cheap, now:** `sensor/temporal.py` — an FNO-rollout that ramps the load (sx,sy from 0→target
  over T steps), renders a marker *video*, and a differentiable temporal slip signal. No new FEM.
- **True GT (Docker, slow):** add `--save-trajectory` to `isaac_extract_shear.py` to store every
  lateral micro-step (`disp_traj[N,T,M,3]`), giving real loading-history GT. Re-run one small combo
  as a smoke (not the full sweep). Then a loading-history FNO variant (extra input = previous-step
  disp) to test the tangential-ceiling lever from [[input-representation-tangential]].
- Verify: temporal preview GIF/montage shows progressive stick→slip; loading-history FNO variant
  vs static FNO on the small trajectory combo (tangential rel-L2 / dir-error).

## 6d — Object geometry + deep contact (Docker/Isaac, riskiest, last)
Generalise beyond the sphere indentor.
- `isaac_extract_shear.py`: `--indentor-geom {sphere,flat,mesh}` + `--indentor-mesh` (MeshFileCfg);
  staged deeper loading; optionally switch to the `isaac-lab-tacex` image (GIPC) for robust deep
  contact. Add a flat-punch + one mesh object as smoke combos.
- Verify (`--smoke` first): a flat-punch and a mesh indentor each produce a stable non-empty frame;
  marker field + sensor render look sensible; then a tiny sweep merged via `aggregate_sweep.py`.

## Critical files
- NEW: `src/novbts/sensor/tactile_env.py`, `sensor/realism.py` (or extend markercam), `sensor/calibration.py`, `sensor/temporal.py`
- MODIFY: `src/novbts/groundtruth/isaac_extract_shear.py` (--save-trajectory, --indentor-geom/--indentor-mesh)
- REUSE: `field2field.py`, `markercam.py`, `diff_policy.py` (PolicyMLP), `aggregate_sweep.py`, `paths.py`
- Docker: `isaac-lab-fem` / `isaac-lab-tacex` via `infra/gen_fem_sweep.sh` pattern (volume `-v $PWD:/work`, chown fix)
- Out: `runs/phase6/`, data under `data/fem/` (trajectory/object variants)

## Verification (overall)
Each milestone has its own check above. End state: `tactile_env` runs a Phase-4 policy to a tactile
target under realistic (noisy) sensor observations, optionally over a temporal rollout, with
object-geometry coverage beyond the sphere — i.e. a usable differentiable VBTS sim framework.

## Notes
- Pure-Python (6a,6b, 6c-cheap) run on the existing `.venv-gate2` + GPU. Isaac pieces (6c-true,6d)
  run via Docker (slow) — smoke-test, don't block on full sweeps.
- Don't auto-rebuild the PDF report ([[feedback-report-updates]]); sync docs only when asked.
