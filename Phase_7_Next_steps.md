# Phase 7 Next Steps: Bottom Boundary Correction for Realistic IPC/UIPC GT

**Date:** 2026-07-01  
**Status:** Root cause confirmed. Corrected BC identified, but full GT regen is gated on robust replicate handling. The all-regime `velocity_tol=3e-4` smoke removes rigid drift but still has high-shear replicate outliers.

---

## 0. Update: Root Cause Confirmed and Candidate BC Selected

The decisive finding is that the original IPC/UIPC Phase 7 issue is a boundary-condition artifact, not a render/camera problem.

Two separate effects had been mixed together:

- The gel bottom was too soft (`SoftPositionConstraint` strength `100`), causing near-rigid tangential pad drift.
- The first hard-bottom test still used a soft kinematic indentor drive (`indentor_constraint_strength=100`), so the sphere missed the requested indentation depth and made hard-bottom look over-stiff.

After separating those effects, the current corrected-BC candidate is:

```bash
--gel-bottom-bc fixed --indentor-constraint-strength 3e4 --velocity-tol 3e-4
```

This candidate passes the BC/localization gate, but not the repeatability gate under plain K=3 averaging.

Artifacts:

- `data/uipc/phase7_bc_sweep/cases.json`
- `data/uipc/phase7_bc_sweep/rows.txt`
- `data/uipc/phase7_bc_sweep/summary.json`
- `data/uipc/phase7_bc_sweep/summary.md`
- per-strength UIPC outputs under `data/uipc/phase7_bc_sweep/strength_*`
- `data/uipc/phase7_drive_probe/summary.json`
- `data/uipc/phase7_drive_probe/summary.md`
- `data/uipc/phase7_fixed_drive_sweep/summary.json`
- `data/uipc/phase7_fixed_drive_sweep/summary.md`
- fixed-drive outputs under `data/uipc/phase7_fixed_drive_sweep/ind_*/combo_000`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE.npz`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0003.npz`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0001.npz`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0003_FULL.npz`

Code added:

- `src/novbts/groundtruth/uniform_shift_diagnostic.py`
- `infra/run_phase7_bc_sweep.sh`

Code updated:

- `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - `--gel-bottom-bc soft|fixed`
  - `--gel-constraint-strength`
  - `--indentor-constraint-strength`
  - all three saved in NPZ provenance
- `src/novbts/groundtruth/aggregate_uipc_replicates.py`
  - preserves the new provenance keys
- `infra/gen_uipc_trajectory_phase7.sh`
  - exposes `GEL_BOTTOM_BC`, `GEL_CONSTRAINT_STRENGTH`, and `INDENTOR_CONSTRAINT_STRENGTH`
- `infra/gen_uipc_sweep.sh`
  - exposes the same BC knobs for corrected production/static GT regen

Sweep cases:

- `normal_mid`: depth `0.55 mm`, no shear
- `partial_mid`: depth `0.55 mm`, drive ratio `0.80`
- `full_mid`: depth `0.55 mm`, drive ratio `1.30`
- `full_max`: depth `0.75 mm`, drive ratio `1.30`

Sweep strengths:

```text
100, 300, 1000, 3000, 10000
```

Main result:

| Setup | Uniform energy | Mean shift | Residual mean | Residual p95 | Depth max | Local `uz` |
|---|---:|---:|---:|---:|---:|---:|
| original-like soft `100` full-max | `99.82%` | `0.309 mm` | `0.013 mm` | `0.016 mm` | `0.683 mm` | `0.085 mm` |
| soft `3000` + indentor `1e4` full-max | `81.95%` | `0.165 mm` | `0.067 mm` | `0.131 mm` | `0.786 mm` | `0.467 mm` |
| fixed + indentor `1e4` full-max | `24.71%` | `0.068 mm` | `0.084 mm` | `0.264 mm` | `0.673 mm` | `0.225 mm` |
| fixed + indentor `3e4` full-max | `25.91%` | `0.086 mm` | `0.105 mm` | `0.328 mm` | `0.749 mm` | `0.272 mm` |
| fixed + indentor `1e5` full-max | `26.08%` | `0.094 mm` | `0.113 mm` | `0.358 mm` | `0.782 mm` | `0.288 mm` |

Interpretation:

- The sweep confirms the root cause across multiple contact cases, not only one frame.
- Strength `100` and `300` leave full-slip cases almost rigid-shift dominated.
- Strength `1000` is the first plausible transition point, but full-slip uniform energy is still `~90--93%`.
- Strength `3000` reduces rigid shift, but a direct `soft3000 + indentor1e4` probe still has `81.95%` uniform energy on the full-max case.
- UIPC's hard vertex route exists through `builtin.is_fixed`; `--gel-bottom-bc fixed` was tested.
- Hard fixed bottom with the old indentor strength `100` was a confounded diagnostic: the driven sphere did not maintain target depth.
- Hard fixed bottom with `indentor_constraint_strength=3e4` matches the full-max requested indentation almost exactly (`0.749 mm` measured vs `0.750 mm` requested) while keeping shear-case uniform energy near `25%`.
- K=3 smoke at `velocity_tol=1e-3` passes the rigid-shift/localization gate but has elevated tangential replicate noise.
- A matched 12-frame probe shows `velocity_tol=3e-4` is currently the best repeatability tradeoff: tangential noise mean improves from `5.82%` to `3.65%`, while `1e-4` lowers the mean to `3.15%` but worsens p95/max.
- The final all-regime smoke at `velocity_tol=3e-4` completed with `72` K=3 averaged frames and all four regimes.
- Localization passes: full-slip uniform energy mean/p95 is `23.52% / 26.06%`, versus the old `~99.8%`.
- Repeatability does not fully pass under plain K=3 averaging: full-slip tangential replicate noise mean/p95/max is `9.77% / 11.83% / 11.87%`; partial has one `20.86%` outlier.
- A targeted K=5 probe on `combo_002` shows robust subset filtering works much better than straight K=5 averaging. For partial/full frames in that combo, straight K5 pairwise tangential noise mean is `9.60%`, while best 4-of-5 is `6.23%` and best 3-of-5 is `4.63%`.
- No full production regen should be launched until robust replicate aggregation is prototyped and re-smoked.

Final smoke result:

| Regime | Count | Uniform energy mean | Uniform energy p95 | Tangential rep-noise mean | Tangential rep-noise p95 |
|---|---:|---:|---:|---:|---:|
| normal | `36` | `4.27%` | `6.29%` | `0.64%` | `1.36%` |
| stick | `17` | `15.71%` | `25.01%` | `2.99%` | `6.81%` |
| partial | `11` | `21.77%` | `24.13%` | `6.00%` | `16.32%` |
| full | `8` | `23.52%` | `26.06%` | `9.77%` | `11.83%` |

Candidate fixed-drive sweep:

| Case | Indentor strength | Uniform energy | Residual mean | Depth max | Local `uz` |
|---|---:|---:|---:|---:|---:|
| `normal_mid` | `3e4` | `5.70%` | `0.022 mm` | `0.559 mm` | `0.177 mm` |
| `partial_mid` | `3e4` | `25.59%` | `0.069 mm` | `0.560 mm` | `0.186 mm` |
| `full_mid` | `3e4` | `24.92%` | `0.093 mm` | `0.560 mm` | `0.188 mm` |
| `full_max` | `3e4` | `25.91%` | `0.105 mm` | `0.749 mm` | `0.272 mm` |

---

## 1. Executive Summary

Phase 7 revealed that the current IPC/UIPC datasets, including the realistic reground dataset, are dominated by near-rigid top-surface tangential translation. This explains why marker animations look like all dots slide together instead of showing localized contact deformation around the sphere.

The issue is not primarily the renderer, camera view, FNO, or temporal model. A single-frame UIPC probe confirmed the likely root cause: the gel bottom is constrained through `SoftPositionConstraint` with strength ratio `100`, which is too soft for a bonded/pinned VBTS gel. Under tangential friction, the whole gel pad can drift, so the top marker field becomes almost a rigid shift.

The next scientific step is to calibrate or replace the bottom boundary condition, then regenerate GT only after the corrected BC passes a small set of physical gates.

---

## 2. Confirmed Evidence

### 2.1 Current Failure Mode

Current realistic UIPC tangential field:

- uniform/rigid component: about `99.8%` of tangential energy
- mean rigid shift: about `0.287 mm`
- local tangential residual: about `0.012 mm`
- visual consequence: marker GIF shows almost uniform sliding

This is not expected for a bonded tactile gel under localized sphere shear.

### 2.2 Probe Setup

One full-slip realistic frame was rerun with different bottom constraint strengths:

- gel: `20 x 20 x 3 mm`
- `gel_res=24`
- sphere radius: `4 mm`
- indentation depth: `0.75 mm`
- shear endpoint: `(-0.390, -0.6755) mm`
- `mu=0.6`, `E=1e5 Pa`
- `eps_velocity=2.5e-5`
- `velocity_tol=1e-3`
- `d_hat=1e-4`
- `contact_resistance=1e9`

Artifacts:

- `data/uipc/phase7_constraint_probe/strength100/uipc_gt_shear.npz`
- `data/uipc/phase7_constraint_probe/strength1e3/uipc_gt_shear.npz`
- `data/uipc/phase7_constraint_probe/strength1e4/uipc_gt_shear.npz`

### 2.3 Probe Results

Tangential field:

| Gel constraint | Uniform energy | Uniform norm | Mean shift | Residual mean | Residual p95 | Max tangential |
|---:|---:|---:|---:|---:|---:|---:|
| `100` | `99.81%` | `99.90%` | `0.287 mm` | `0.012 mm` | `0.016 mm` | `0.332 mm` |
| `1e3` | `87.88%` | `93.74%` | `0.051 mm` | `0.018 mm` | `0.023 mm` | `0.112 mm` |
| `1e4` | `45.06%` | `67.12%` | `0.008 mm` | `0.006 mm` | `0.015 mm` | `0.070 mm` |

Normal field side effect:

| Gel constraint | Depth p95 | Depth max | Local center-edge `uz` |
|---:|---:|---:|---:|
| `100` | `0.616 mm` | `0.643 mm` | `0.085 mm` |
| `1e3` | `0.193 mm` | `0.294 mm` | `0.142 mm` |
| `1e4` | `0.057 mm` | `0.166 mm` | `0.067 mm` |

Interpretation:

- Increasing bottom constraint strength collapses the rigid-shift artifact.
- Therefore the soft bottom constraint is a real root cause.
- But `1e4` is not automatically a production fix because it also changes normal indentation too much.

---

## 3. Current Code State

Implemented:

- `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - added `--gel-constraint-strength`
  - added `--indentor-constraint-strength`
  - both default to `100`
  - both are saved in NPZ provenance
- `src/novbts/groundtruth/aggregate_uipc_replicates.py`
  - preserves the new provenance keys through replicate/frame aggregation
- `src/novbts/sensor/temporal.py`
  - records trajectory semantics and field stats
  - uses auto-scaled camera by default
- `src/novbts/sensor/temporal_compare.py`
  - uses the same auto-scaled camera logic

Validation already run:

```bash
rtk proxy .venv-gate2/bin/python -m py_compile \
  src/novbts/groundtruth/tacex_uipc_extract_shear.py \
  src/novbts/groundtruth/aggregate_uipc_replicates.py \
  src/novbts/sensor/temporal.py \
  src/novbts/sensor/temporal_compare.py
```

---

## 4. Consequence for Existing Results

The existing realistic dataset:

`data/uipc/shear_res24_avg_swept_REALISTIC.npz`

is still internally consistent as a machine-learning benchmark against its own GT. The FNO metrics remain valid as "FNO reproduces this GT."

However, it should not be used as strong physical evidence that the pipeline captures localized tangential VBTS shear. The current tangential/local channel is contaminated by a boundary-condition artifact.

Paper/report guidance:

- Keep current results only if framed as provisional/current-GT surrogate metrics.
- Do not claim physically faithful localized shear/contact imprint from the current IPC tangential field.
- Do not rely on marker GIFs as evidence for sphere indentation or local shear.
- The normal channel may still be more usable, but it must be rechecked after BC correction because normal response changes with constraint strength.

---

## 5. Recommended Next Steps

### Step 1: Boundary-Condition Calibration Sweep

Status: first pass completed. Do not launch full production regen yet.

Completed strengths:

```text
100, 300, 1e3, 3e3, 1e4
```

Completed cases:

1. normal-only, moderate depth
2. partial-slip, moderate depth
3. full-slip, moderate depth
4. full-slip, max depth

Completed before full production acceptance:

- add one smaller-radius and one larger-radius case
- run K=3 replicate check at `fixed + indentor 3e4`
- verify the corrected field no longer looks like bulk sliding in uniform-shift diagnostics

Decision target remains:

- tangential uniform energy should no longer be near `99--100%`
- normal indentation should remain physically plausible
- residual/local tangential signal should not vanish
- solver/replicate aggregation should be stable enough at `gel_res=24`

### Step 2: Hard Dirichlet With Corrected Indentor Drive

Status: corrected BC selected; repeatability still needs robust aggregation.

UIPC/TacEx supports a hard fixed-vertex route through `builtin.is_fixed`, and the driver now exposes it as:

```bash
--gel-bottom-bc fixed --indentor-constraint-strength 3e4
```

The earlier hard-bottom result that nearly eliminated visible normal response was confounded by the indentor drive being left at strength `100`. With `indentor_constraint_strength=3e4`, the fixed-bottom full-max probe reaches `0.749 mm` measured peak depth for a `0.750 mm` target and keeps full-slip uniform energy near `26%`.

Questions to answer:

- Which robust replicate policy should be used for high-shear frames?
- Is K=5 with medoid/best-4 filtering enough across representative combos?
- Does the corrected trajectory render show localized motion without visual exaggeration?
- Do downstream metrics remain sane after regenerating a small corrected dataset?

### Step 3: Pick a Production Replicate Policy

Current candidate:

```bash
GEL_BOTTOM_BC=fixed INDENTOR_CONSTRAINT_STRENGTH=3e4 VELOCITY_TOL=0.0003
```

BC is selected, but production should not use plain K=3 averaging for high-shear tangential GT. Prototype one robust policy first:

- K=5 raw UIPC reps
- compute pairwise field distances per frame, preferably tangential and overall
- reject one outlier or choose a medoid/best-4 subset
- average the retained reps
- store robust aggregation provenance: raw K, kept K, rejected reps, pairwise distances

Minimum acceptance gates:

- tangential rigid-shift artifact clearly reduced
- normal indentation not implausibly over-stiff
- robust replicate noise acceptable in partial/full frames
- res24 remains stable enough
- all four modes still represented: normal/stick/partial/full

### Step 4: Robust Production-Shaped Smoke

Before full regen, run a small sweep with the chosen BC and robust aggregation:

- at least `1--3` combos
- K=5 raw reps
- robust best-4 or medoid aggregation
- include normal/stick/partial/full
- aggregate and inspect provenance
- render marker/depth diagnostics

Suggested raw smoke command:

```bash
rtk proxy env \
  SWEEP_DIR=data/uipc/sweep_realistic_bc_robust_smoke \
  OUT_DATA=data/uipc/shear_res24_avg_swept_REALISTIC_BC_ROBUST_SMOKE_RAW.npz \
  GEL_BOTTOM_BC=fixed \
  INDENTOR_CONSTRAINT_STRENGTH=3e4 \
  TEST_SIZE=9 \
  GEL_RES=24 EPS_VELOCITY=0.000025 D_HAT=0.0001 \
  CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.0003 \
  bash infra/gen_uipc_sweep.sh 3 12 5
```

Acceptance:

- output NPZ contains `gel_constraint_strength`
- output NPZ contains `gel_bottom_bc=fixed`
- output NPZ contains `indentor_constraint_strength=30000`
- robust provenance records raw K and kept reps
- local tangential residual is visible above previous `~0.012 mm` artifact level
- uniform fraction is not near `100%`
- normal channel remains plausible
- partial/full tangential replicate noise is materially lower than plain K=3

### Step 5: Full GT Regeneration

Only after robust-smoke acceptance, regenerate a corrected realistic dataset, ideally with a new filename rather than overwriting the current final dataset.

Suggested filename:

```text
data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz
```

Keep the current file as provenance:

```text
data/uipc/shear_res24_avg_swept_REALISTIC.npz
```

Suggested full regen command after robust smoke acceptance:

```bash
rtk proxy nohup env \
  SWEEP_DIR=data/uipc/sweep_realistic_bc \
  OUT_DATA=data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz \
  GEL_BOTTOM_BC=fixed \
  INDENTOR_CONSTRAINT_STRENGTH=3e4 \
  GEL_RES=24 EPS_VELOCITY=0.000025 D_HAT=0.0001 \
  CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.0003 \
  bash infra/gen_uipc_sweep.sh 63 40 5 \
  > logs/uipc_sweep_realistic_bc.log 2>&1 &
```

This command still needs a robust aggregation step before it is production-safe.

### Step 6: Downstream Rerun

After corrected GT exists:

1. rerun Phase 3 FNO/MLP benchmark
2. rerun VBTS baselines
3. rerun policy servo if action gradients depend on tangential field
4. rerun sensor build/inversion
5. rerun tactile env/gradcheck
6. regenerate paper figures
7. update paper/report metrics
8. rerun final verifier, with new BC provenance checks

---

## 6. Suggested New Verifier Gates

Add checks to `infra/verify_realistic_reground.py` or a new BC-specific verifier:

- dataset includes `gel_constraint_strength`
- selected BC equals documented production value
- uniform tangential energy is below a chosen threshold on sampled full-slip frames
- current GT is newer than corrected BC probe report
- paper does not claim localized tangential fidelity unless corrected BC dataset is used
- legacy/current datasets are not silently overwritten

Possible diagnostic thresholds for discussion:

```text
uniform_energy_full_slip_mean < 90%
uniform_energy_full_slip_p95  < 97%
residual_tangential_mean      > previous artifact floor
gel_bottom_bc                 == fixed
indentor_constraint_strength  == 30000
velocity_tol                  == 0.0003
raw_replicates                >= 5
kept_replicates               >= 4
```

These are not final scientific thresholds. They are starting gates to prevent another near-rigid-shift dataset from passing unnoticed.

---

## 7. Immediate Action List

1. Prototype robust replicate aggregation on the existing K=5 `combo_002` probe.
2. Run a K=5 robust smoke across representative combos.
3. Run `uniform_shift_diagnostic.py` and replicate-noise diagnostics on the robust output.
4. Render corrected Phase 7 trajectory with the same BC.
5. If robust gates pass, launch full corrected GT production under a new filename.
6. Rerun downstream and paper after corrected GT is accepted.

---

## 8. Bottom Line

The Phase 7 problem is now a GT boundary-condition problem, not a visualization problem.

The current realistic IPC/UIPC GT should be considered provisional for localized tangential VBTS evidence. The corrected BC is identified, but the next decisive work is robust replicate aggregation for high-shear UIPC outliers; only then should corrected GT be regenerated and downstream metrics rerun.
