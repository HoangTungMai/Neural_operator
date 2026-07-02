# PROGRESS: realistic VBTS gel geometry reground

## Latest checkpoint (2026-07-01 Phase7 vtol0003 full smoke + K5 outlier audit +07)

All-regime smoke with corrected boundary conditions completed, but it should NOT
be treated as full production acceptance yet.

Corrected candidate:

```bash
GEL_BOTTOM_BC=fixed INDENTOR_CONSTRAINT_STRENGTH=3e4 VELOCITY_TOL=0.0003
```

Smoke artifact:

- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0003_FULL.npz`
- `N=72`, K=3 averaged frames
- modes: normal=36, stick=17, partial=11, full=8
- provenance: `gel_bottom_bc=fixed`, `indentor_constraint_strength=30000`,
  `velocity_tol=0.0003`, `gel_res=24`, bilinear marker sampling

Rigid-shift/localization result:

- normal uniform energy mean/p95: `4.27% / 6.29%`
- stick uniform energy mean/p95: `15.71% / 25.01%`
- partial uniform energy mean/p95: `21.77% / 24.13%`
- full uniform energy mean/p95: `23.52% / 26.06%`
- full residual tangential mean: `0.066 mm`

Conclusion on BC:

- The old soft-BC artifact (`~99.8--99.9%` uniform energy) is removed.
- The corrected BC is physically much better for localized tangential GT.

Repeatability result:

- all-frame tangential replicate noise mean/p95/max: `3.03% / 11.25% / 20.86%`
- normal mode tangential noise mean/p95/max: `0.64% / 1.36% / 2.13%`
- stick: `2.99% / 6.81% / 7.10%`
- partial: `6.00% / 16.32% / 20.86%`
- full: `9.77% / 11.83% / 11.87%`

Outlier audit:

- worst frame: `combo_002/frame_009`, partial, `R=6 mm`, depth `0.316 mm`,
  shear `0.533 mm`, tangential noise `20.86%`
- raw reps show rep 2 and rep 3 are close (`4.50%` tangential pair distance),
  while rep 1 is far from both (`28.18%` and `29.90%`)
- other full-slip outliers are more distributed and are not explained by one
  single bad replicate

Targeted K=5 probe:

- Added reps 4 and 5 for `combo_002` only. The final smoke NPZ was NOT
  re-aggregated; it remains K=3.
- For combo 002, straight K=5 improves but does not fully solve high-shear noise:
  - partial/full mean pair tangential noise: K3 `12.23%` -> K5 `9.60%`
  - full mean pair tangential noise: K3 `9.36%` -> K5 `8.09%`
- Robust subset behavior is much better:
  - best 3-of-5 partial/full mean: `4.63%`
  - best 4-of-5 partial/full mean: `6.23%`
  - best 3-of-5 full mean: `4.78%`
  - best 4-of-5 full mean: `6.71%`

Current conclusion:

- BC fix is valid.
- Remaining blocker is UIPC stochasticity/outlier replicates in high-shear
  contact.
- Do not launch full corrected GT with plain K=3 averaging if the tangential GT
  will be used as strong physical evidence.

Next action:

1. Implement or prototype a robust replicate aggregator, likely K=5 with
   medoid/best-4 filtering by pairwise field distance.
2. Run a small robust K=5 smoke across representative combos.
3. Launch full corrected GT only after the robust smoke passes repeatability.

---

## Latest checkpoint (2026-07-01 Phase7 fixed-bottom drive calibration +07)

Phase 7 root cause is now separated cleanly from render/camera:

- Original realistic IPC/UIPC shear field was dominated by bottom-BC rigid drift.
- Soft gel bottom `SoftPositionConstraint` at strength `100` causes full-slip
  uniform tangential energy near `99.8--99.9%`.
- The first hard-bottom test was confounded: the gel bottom was fixed, but the
  indentor drive stayed at strength `100`, so the sphere missed its target depth
  and made hard-bottom look artificially over-stiff.

New/updated artifacts:

- `data/uipc/phase7_drive_probe/summary.md`
- `data/uipc/phase7_fixed_drive_sweep/summary.md`
- `data/uipc/phase7_fixed_drive_sweep/ind_{1e4,3e4,1e5}/combo_000/...`
- regenerated `data/uipc/phase7_bc_sweep/summary.md`

Key result:

- `soft3000 + indentor1e4` full-max still has `81.95%` uniform energy.
- `fixed + indentor1e4` full-max drops to `24.71%` uniform energy but only
  reaches `0.673 mm` peak depth for a `0.750 mm` target.
- `fixed + indentor3e4` full-max drops to `25.91%` uniform energy and reaches
  `0.749 mm` peak depth for the `0.750 mm` target.
- `fixed + indentor1e5` also works but overshoots full-max to `0.782 mm`.

Current candidate corrected setup:

```bash
GEL_BOTTOM_BC=fixed INDENTOR_CONSTRAINT_STRENGTH=3e4 VELOCITY_TOL=0.0003
```

Code updates:

- `src/novbts/groundtruth/uniform_shift_diagnostic.py`
  now reports gel BC, gel strength, indentor strength, depth max, and depth p95.
- `infra/run_phase7_bc_sweep.sh` exposes indentor drive strength.
- `infra/gen_uipc_trajectory_phase7.sh` exposes BC env knobs.
- `infra/gen_uipc_sweep.sh` exposes BC env knobs for corrected production GT.
- Syntax checks passed:
  - `bash -n infra/run_phase7_bc_sweep.sh`
  - `bash -n infra/gen_uipc_trajectory_phase7.sh`
  - `bash -n infra/gen_uipc_sweep.sh`
  - `py_compile` for UIPC driver/aggregator/diagnostic.

Next action:

K=3 smoke status:

- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE.npz`
  - 40 averaged frames, K=3
  - all regimes present: mode counts `{0:4, 1:17, 2:11, 3:8}`
  - provenance: `gel_bottom_bc=fixed`, `indentor_constraint_strength=30000`,
    `marker_sampling=bilinear`
  - full-slip uniform energy mean/p95: `23.78% / 26.29%`
  - full-slip residual tangential mean: `0.069 mm`
  - tangential replicate noise at `velocity_tol=1e-3`: mean `5.66%`, p95
    `12.33%`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0003.npz`
  - 12-frame matched subset at `velocity_tol=3e-4`
  - matched-subset tangential noise improved from `5.82%` to `3.65%`
  - p95 improved from `10.60%` to `8.17%`
- `data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0001.npz`
  - mean noise improved further to `3.15%`, but p95/max worsened
    (`9.74%` / `15.51%`), so it is less attractive than `3e-4`

Next action:

1. Run the full all-regime smoke at `VELOCITY_TOL=0.0003`:

```bash
rtk proxy env \
  SWEEP_DIR=data/uipc/sweep_realistic_bc_smoke_vtol0003_full \
  OUT_DATA=data/uipc/shear_res24_avg_swept_REALISTIC_BC_SMOKE_VTOL0003_FULL.npz \
  GEL_BOTTOM_BC=fixed \
  INDENTOR_CONSTRAINT_STRENGTH=3e4 \
  TEST_SIZE=9 \
  GEL_RES=24 EPS_VELOCITY=0.000025 D_HAT=0.0001 \
  CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.0003 \
  bash infra/gen_uipc_sweep.sh 3 12 3
```

2. Add normal top-up or run a production-shaped all-regime smoke if mode 0 is
   missing.
3. Run `uniform_shift_diagnostic.py`, inspect provenance, and render corrected
   Phase 7 trajectory.
4. Only after smoke acceptance, launch full corrected GT as
   `data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz`.

## Latest checkpoint (2026-07-01 Phase7 hard-bottom test +07)

Continued from BC sweep by checking whether UIPC has a true fixed-vertex route.

Findings:

- UIPC/TacEx does expose hard fixed FEM vertices through `builtin.is_fixed`.
- `src/novbts/groundtruth/tacex_uipc_extract_shear.py` now has:
  - `--gel-bottom-bc soft` (default, historical behavior)
  - `--gel-bottom-bc fixed` (sets `builtin.is_fixed=1` on bottom vertices)
- `gel_bottom_bc` is saved into NPZ provenance and propagated by
  `aggregate_uipc_replicates.py`.
- `infra/run_phase7_bc_sweep.sh` now includes fixed-bottom by default
  (`INCLUDE_FIXED=1`).

Fixed-bottom artifacts:

- one-frame probe:
  `data/uipc/phase7_constraint_probe/fixed/uipc_gt_shear.npz`
- 4-case sweep:
  `data/uipc/phase7_bc_sweep/fixed/combo_000/...`
- updated summaries:
  `data/uipc/phase7_bc_sweep/summary.json`
  `data/uipc/phase7_bc_sweep/summary.md`

Fixed-bottom sweep result:

- It removes rigid shift:
  - partial/full uniform energy `~15--18%`
  - full mean shift only `~0.003 mm`
- But it over-stiffens the current setup:
  - full-mid/full-max depth p95 only `0.003--0.005 mm`
  - local `uz` only `0.010--0.014 mm`

Conclusion:

- Hard fixed bottom confirms the BC mechanism, but is not production-ready under
  the current material/contact setup.
- Current candidate soft band remains `1000--3000`, but it still changes normal
  response and needs K=3/mesh sanity before any regen.
- Next useful direction: either implement/test a more nuanced bonded-bottom BC
  (for example tangentially fixed with calibrated normal compliance, if UIPC can
  support it) or run K=3 production-shaped smoke at soft `1000`/`3000` before
  choosing a corrected provisional GT.
- `Phase_7_Next_steps.md` updated accordingly.

## Latest checkpoint (2026-07-01 Phase7 BC sweep completed +07)

BC confirmation sweep completed after the single-frame probe.

New reusable tools:

- `src/novbts/groundtruth/uniform_shift_diagnostic.py`
  - computes tangential uniform/rigid fraction, residual tangential metrics,
    depth p95/max, and local center-edge `uz`
- `infra/run_phase7_bc_sweep.sh`
  - runs 4 representative UIPC cases over bottom constraint strengths
  - writes `summary.json` and `summary.md`

Artifacts:

- `data/uipc/phase7_bc_sweep/cases.json`
- `data/uipc/phase7_bc_sweep/rows.txt`
- `data/uipc/phase7_bc_sweep/summary.json`
- `data/uipc/phase7_bc_sweep/summary.md`
- per-strength outputs under `data/uipc/phase7_bc_sweep/strength_*`

Sweep:

- strengths: `100`, `300`, `1000`, `3000`, `10000`
- cases:
  - `normal_mid`: depth `0.55 mm`, no shear
  - `partial_mid`: depth `0.55 mm`, drive ratio `0.80`
  - `full_mid`: depth `0.55 mm`, drive ratio `1.30`
  - `full_max`: depth `0.75 mm`, drive ratio `1.30`

Key results:

- `strength=100` reproduces the artifact:
  - partial/full uniform energy `99.77--99.88%`
  - full mean shift `0.288--0.309 mm`
  - residual mean only `0.010--0.013 mm`
- `strength=300` still leaves full cases rigid dominated:
  - full uniform energy `98.58--98.89%`
- `strength=1000` is the first transition point:
  - partial uniform energy `83.46%`
  - full uniform energy `90.18--92.81%`
  - residual mean `0.014--0.018 mm`
  - depth p95 drops to `0.149--0.195 mm`
- `strength=3000` reduces rigid shift more:
  - full uniform energy `73.18--78.71%`
  - mean shift `0.021--0.022 mm`
  - but depth p95 drops to `0.084--0.112 mm`
- `strength=10000` is very stiff:
  - full uniform energy `54.32--55.22%`
  - depth p95 only `0.042--0.056 mm`

Conclusion:

- Root cause is now confirmed across multiple cases: bottom soft constraint
  strength `100` produces near-rigid tangential top-surface drift.
- No production BC chosen yet.
- Candidate soft-constraint band is roughly `1000--3000`, but this also changes
  normal response substantially.
- Next decisive step is to test a true hard-bottom/Dirichlet implementation or
  better UIPC constraint formulation, then compare it against the `1000--3000`
  band before any corrected GT production regen.
- `Phase_7_Next_steps.md` was updated with the completed sweep and current plan.

## Latest checkpoint (2026-07-01 Phase7 bottom-constraint probe +07)

User's diagnosis was confirmed by a cheap UIPC single-frame probe.

Problem tested:

- Current IPC/UIPC tangential fields are dominated by near-rigid top-surface
  translation instead of localized contact deformation.
- Suspected cause: gel bottom uses UIPC `SoftPositionConstraint` with strength
  ratio `100`, so the pad can still drift under tangential friction.

Implementation changes:

- `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - added `--gel-constraint-strength`, default `100`
  - added `--indentor-constraint-strength`, default `100`
  - saved both values into NPZ provenance
- `src/novbts/groundtruth/aggregate_uipc_replicates.py`
  - preserves those provenance keys through replicate/frame aggregation
- Python compile passed for both files.

Probe setup:

- one full-slip realistic frame matching Phase7 frame 11:
  - gel `20 x 20 x 3 mm`, `gel_res=24`
  - `depth=0.75 mm`, `R=4 mm`, `mu=0.6`, `E=1e5 Pa`
  - shear endpoint `(-0.390, -0.6755) mm`, drive ratio `1.3`
  - solver knobs unchanged: `eps_velocity=2.5e-5`, `velocity_tol=1e-3`,
    `d_hat=1e-4`, `contact_resistance=1e9`
- artifacts:
  - `data/uipc/phase7_constraint_probe/strength100/uipc_gt_shear.npz`
  - `data/uipc/phase7_constraint_probe/strength1e3/uipc_gt_shear.npz`
  - `data/uipc/phase7_constraint_probe/strength1e4/uipc_gt_shear.npz`

Results on top-marker tangential field:

| Gel constraint | uniform energy | uniform norm | mean shift | residual mean | residual p95 | max tangential |
|---:|---:|---:|---:|---:|---:|---:|
| `100` | `99.81%` | `99.90%` | `0.287 mm` | `0.012 mm` | `0.016 mm` | `0.332 mm` |
| `1e3` | `87.88%` | `93.74%` | `0.051 mm` | `0.018 mm` | `0.023 mm` | `0.112 mm` |
| `1e4` | `45.06%` | `67.12%` | `0.008 mm` | `0.006 mm` | `0.015 mm` | `0.070 mm` |

Normal field side effect:

| Gel constraint | depth p95 | depth max | local center-edge `uz` |
|---:|---:|---:|---:|
| `100` | `0.616 mm` | `0.643 mm` | `0.085 mm` |
| `1e3` | `0.193 mm` | `0.294 mm` | `0.142 mm` |
| `1e4` | `0.057 mm` | `0.166 mm` | `0.067 mm` |

Conclusion:

- The soft bottom constraint is a real root cause of the rigid-shift artifact.
- Increasing bottom strength collapses the uniform rigid shift by an order of
  magnitude, so the current realistic IPC/UIPC GT tangential channel is not a
  trustworthy localized shear/contact-deformation target.
- `1e4` is not automatically the production fix: it makes the pad much stiffer
  and changes normal indentation too much.
- Next scientific step should be a small calibration sweep over bottom boundary
  strength and/or a true hard Dirichlet implementation, then regenerate GT only
  after choosing a physically defensible BC.
- Existing realistic downstream metrics remain internally consistent
  FNO-vs-current-GT, but tangential/local VBTS evidence should be treated as
  suspect until GT is regrounded with corrected bottom BC.

## Latest checkpoint (2026-07-01 review PASS + state summary +07)

Realistic-geometry reground (Phases 0-3) reviewed end-to-end and confirmed
complete. Not just verifier-trusted: paper text re-read and cross-checked against
regenerated JSONs (sensor cosine/round-trip, class counts, RQ2/RQ3 all match).

State now:

- `infra/verify_realistic_reground.py`: `REALISTIC_REGROUND_ACCEPTANCE_OK`
  (dataset / downstream JSON / figures / paper / legacy-SHA all pass).
- No uipc systemd service active; `0` docker containers. Nothing running.
- Legacy dataset intact:
  `data/uipc/shear_res24_avg_swept.npz`
  SHA-256 `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`.
- Final realistic GT: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  N=`2520`, train/test `2120/400`, modes `425/712/895/488`, K=3, gel
  `20 x 20 x 3 mm`, bilinear sampling.
- Headline: FNO rel-L2 `0.041` / dir `1.6 deg`, `12.14x` vs MLP; RQ3 speedup
  `83204x` vs fair single solve; RQ5 cosine per regime
  `-0.396/0.559/0.918/0.967`; inversion `15.51% / 3.79 deg`.
- Paper `docs/kse2026/main.tex` builds clean (0 undefined refs); false Hertz
  `1.3%` validator claim dropped, flat-punch OOD dropped, per-regime RQ5,
  rewritten Limitations.

Phase 7 (temporal/loading-history) status: SUPPLEMENTAL / DIAGNOSTIC ONLY, not
paper evidence. The realistic thin-gel marker GIF has genuinely small local
surface contrast (`~0.085 mm` center-edge normal, `~0.012 mm` residual
tangential); camera auto-rescaled to the geometry (`working_dist ~9.9 mm`) but
the imprint stays weak by construction. Use depth/residual diagnostics + metrics,
not the marker-only GIF, if Phase 7 is shown at all.

Working tree (branch `phase4-diff-policy`), uncommitted:

- `PROGRESS.md`
- `infra/gen_uipc_trajectory_phase7.sh`
- `src/novbts/sensor/temporal.py`
- `src/novbts/sensor/temporal_compare.py`

Remaining items (USER-owned, not delegated):

1. Commit the dirty working tree when ready.
2. Paper author de-anonymization / double-blind decision.
3. Funding acknowledgment.

Deadline KSE2026: `2026-07-15`.

## Latest checkpoint (2026-06-30 Phase7 root-cause confirmation +07)

User reported that the updated Phase7 images still do not change much. Confirmed
the real issue with direct field and image audits.

What is true:

- The camera correction is real but limited:
  - old fixed camera `working_dist=50 mm`
  - auto realistic camera `working_dist=9.9 mm`
  - marker-flow increase from `~2.0--2.3 px` to `~3.1--3.3 px`
  - image difference between old/new `temporal_video.png`: mean absolute pixel
    diff `0.027`, p95 `0.180`
- The physical field still has weak *local* shape:
  - final local center-edge normal contrast only `0.085--0.086 mm`
  - residual non-rigid tangential mean only `~0.012 mm`
  - most visible `z` camera motion comes from broad/near-uniform top-surface
    vertical displacement, not a localized sphere imprint
- The trajectory itself is not a press-in trajectory:
  - UIPC `_trajectory_snap_indices()` sets `f=0` to `shear_start - 1`
  - therefore `disp_traj[:,0]` is already the settled pure-normal pressed state
  - `f=0..1` only records lateral/shear loading after press
  - local normal contrast is basically flat over trajectory:
    `~0.087 mm` at `f=0` to `~0.085 mm` at `f=1`

Implementation update:

- `src/novbts/sensor/temporal.py` now records:
  - `trajectory_semantics`
  - per-load-mode `field_stats.by_frac`
  - final z-only and xy-only marker-flow decomposition
- Re-rendered `runs/phase7/temporal.*`.

Root-cause conclusion:

- The updated images are not dramatically different because Phase7 is currently
  visualizing a shear-after-press path whose normal imprint is already present at
  frame 0 and is spatially weak/local-low-contrast.
- To show a sphere visibly pressing into gel, we need a different trajectory
  capture that includes press snapshots from rest through indentation, not just
  lateral shear snapshots after normal loading.
- To make the existing Phase7 useful, present it as loading-history/shear-path
  evidence and use depth/residual diagnostics for the normal field.

## Latest checkpoint (2026-06-30 Phase7 camera-scale correction +07)

User correctly pointed out that the camera/view should change with the realistic
gel scale. Confirmed: Phase 7 temporal renderer was still using the old absolute
`working_dist=50 mm`, inherited from the large old geometry. Because
`PinholeCamera.from_gel(...)` auto-fits the rest marker footprint, this did not
change the rest FOV, but it strongly suppressed `uz`-driven perspective/magnify
motion. Normal-depth sensitivity scales roughly with `|uz| / working_dist`.

Fix applied:

- `src/novbts/sensor/temporal.py`
  - `--working-dist` now defaults to auto rather than fixed `0.05`.
  - auto distance is `working_dist_ratio * marker_half_extent`, default ratio
    `1.1`.
  - This preserves the old large-gel view (`45 mm * 1.1 ~= 50 mm`) while giving
    the realistic Phase7 gel `9 mm * 1.1 ~= 9.9 mm`.
  - `runs/phase7/temporal.json` now records the camera block.
- `src/novbts/sensor/temporal_compare.py` uses the same auto-scaled camera so
  GT-vs-FNO temporal comparison matches the GIF view.
- Re-rendered `runs/phase7/temporal.*` from
  `data/uipc/trajectory_phase7_fullslip/shear_res24_traj_REALISTIC.npz`.

New Phase7 camera/render numbers:

- auto camera: marker half extent `9.0 mm`, working distance `9.9 mm`
- final mean marker flow increased from the stale-camera `~2.0--2.3 px` to
  `3.12--3.26 px`
- flow decomposition at final full-slip frames:
  z-perspective-only `2.51--2.62 px`, xy-only `1.86--2.16 px`
- physical field magnitude remains small:
  local center-edge normal contrast `0.085--0.086 mm`,
  detrended tangential residual mean `~0.012 mm`

Interpretation update:

- The previous "field is physically small" conclusion still holds in mm-space.
- But the previous marker GIF was also under-sensitive because the camera had
  not been rescaled with the geometry.
- For Phase7 visuals, use the auto-scaled camera outputs now in `runs/phase7/`.

## Latest checkpoint (2026-06-30 Phase7 deformation visibility decision +07)

Phase 7 issue investigated: apparent deformation remains too low throughout the
trajectory because the realistic UIPC field genuinely has very small local
surface contrast, not because the temporal renderer/model lost the signal.

Fresh audit on
`data/uipc/trajectory_phase7_fullslip/shear_res24_traj_REALISTIC.npz`:

- selected full-slip frames use `depth=0.75 mm`, `R=4.0 mm`,
  `shear=0.78 mm`
- marker stream final mean flow is visible but modest: `1.98--2.28 px`
- local normal imprint is tiny:
  `center-minus-edge -uz = 0.085--0.086 mm`
- detrended tangential residual is smaller:
  mean `0.012 mm`, max `0.022--0.045 mm`

Comparison points:

- production realistic static dataset max local normal contrast is only about
  `0.178 mm`, residual lateral mean about `0.023 mm`
- old PhysX dataset had several-mm local normal contrast and
  `~0.34--0.79 mm` residual lateral means, so its marker GIF looked like a
  strong sphere press for scale reasons
- scaled/stress pilots (`R=8 mm`, `depth=1.2 mm`) improve visibility only to
  about `0.19 mm` local normal contrast and `0.025--0.031 mm` residual lateral
  mean; they are diagnostic visuals, not paper evidence

Implementation update:

- `src/novbts/sensor/temporal.py` now writes quantitative `field_stats` into
  `runs/phase7/temporal.json` alongside the GIF/figures.
- Re-rendered Phase7 full-slip temporal artifacts in `runs/phase7/`.

Decision:

- Do not spend more time trying to make the realistic marker-only GIF look like
  the old PhysX sphere press; that would require non-realistic geometry/scale or
  visual exaggeration.
- For Phase 7 evidence, use `temporal_depth_residual.png`,
  `phase7_field_diagnostic.png`, and `temporal.json` metrics.
- If a visibly intuitive animation is needed, label it explicitly as a
  diagnostic/stress visualization and keep it separate from paper evidence.

## Latest checkpoint (2026-06-30 Phase7 temporal visual audit +07)

User compared old PhysX trajectory figures with new UIPC Phase7 and correctly
flagged that the new marker GIF does not visually look like a sphere pressing
into gel.

Audit result:

- This is not an FNO/temporal-model failure. The UIPC Phase7 GT contains normal
  indentation in `uz`, but at realistic scale the visible marker image is
  dominated by nearly uniform in-plane shear translation.
- Old PhysX example frame `data/fem/shear_fine_swept_normaug.npz[1643]`:
  depth `6.73 mm`, radius `24.7 mm`, shear `6.76 mm`,
  local center-edge normal contrast `3.57 mm`,
  detrended lateral residual mean/max `0.62/2.12 mm`.
- New UIPC Phase7 full-slip example
  `data/uipc/trajectory_phase7_fullslip/shear_res24_traj_REALISTIC.npz[9]`:
  depth `0.75 mm`, radius `4.0 mm`, shear `0.78 mm`,
  local center-edge normal contrast `0.075 mm`,
  detrended lateral residual mean/max `0.012/0.021 mm`.
- Therefore the old PhysX dot render looked like a strong sphere press because
  the physical scale and local residual deformation were much larger. The
  realistic UIPC dot render should not be expected to match that look.

Pipeline update:

- `infra/gen_uipc_trajectory_phase7.sh` defaults were strengthened for a
  full-slip Phase7 pilot:
  - default depths `0.35/0.55/0.75 mm`
  - default drive ratios `0.30/0.80/1.30`
  - env overrides `DEPTH_LEVELS`, `DRIVE_LEVELS`
- `src/novbts/sensor/temporal.py` now writes extra visual diagnostics:
  - `runs/phase7/temporal_depth_residual.png`
  - `runs/phase7/phase7_field_diagnostic.png`
- Re-rendered Phase7 temporal outputs from
  `data/uipc/trajectory_phase7_fullslip/shear_res24_traj_REALISTIC.npz`.
- Additional visibility tests:
  - Rendered production realistic max-contact frames to
    `runs/phase7/realistic_max_contact_visibility.png`.
  - Ran one diagnostic-only UIPC stress case, not paper evidence:
    `data/uipc/phase7_visibility_stress/r8_d12_normal/uipc_gt_shear.npz`
    with `R=8 mm`, `depth=1.2 mm`, normal-only.
  - Stress case increased local normal contrast from `0.055 mm` (production max
    normal, `R=6 mm`, `depth=0.75 mm`) to `0.166 mm`, and residual lateral
    mean/max from `0.008/0.009 mm` to `0.025/0.029 mm`.
  - Stress comparison figure:
    `runs/phase7/stress_contact_visibility.png`.

Interpretation for paper:

- Do not use the marker-only Phase7 GIF as primary evidence for sphere
  indentation.
- If Phase7 is included, show the normal-depth/residual diagnostic or frame it
  as a supplemental temporal/path sanity check, not a main benchmark result.

## Latest checkpoint (2026-06-30 GPU rerun +07)

User noticed sandbox Python was not using GPU. Confirmed:

- sandbox torch CUDA: unavailable
- `rtk proxy .venv-gate2/bin/python`: CUDA available on NVIDIA RTX 2000 Ada
- stopped the accidental CPU benchmark and reran runnable downstream stages via
  `rtk proxy`

GPU-canonical downstream refresh:

- `runs/phase3_fem/benchmark.json`
  - FNO rel-L2 overall `0.041`, direction `1.6 deg`
  - MLP rel-L2 overall `0.501`, direction `83.9 deg`
  - FNO/MLP rel-L2 advantage `12.14x`
  - RQ3 FNO `7803 fps`, fair GT solver `0.094 fps`, K3 target `0.031 fps`
  - speedup vs fair single solve `82827x`
- `runs/phase3_fem/vbts_baselines.json`
  - U-Net updated to rel-L2 `0.067`, direction `3.5 deg`, FNO advantage `1.63x`
- `runs/phase4/policy_servo.json`
  - autograd final `5.70e-10`, ES final `6.21e-10`
  - autograd reaches target in `21` fwd queries vs ES mean `1771`
  - wall `13.21 s` vs `317.75 s` (`24.1x`)
- `runs/phase5/*`
  - sensor build/inverse regenerated on realistic data with current renderer
- `runs/phase6/env_demo.json`
  - policy closes `99.93%` random-to-oracle reward gap
  - finite-difference diagnostic rel error `5.21%`, `passed=false`
  - paper/report now phrase this as gradient-flow sanity check, not formal pass

Figures regenerated from the GPU-current runs:

- `docs/kse2026/figs/fidelity_speed.png`
- `docs/kse2026/figs/policy_servo_curve.png`
- `docs/kse2026/figs/sensor_gt_vs_fno.png`

Phase 7 trajectory/loading-history design started:

- Added UIPC trajectory support to
  `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - `--save-trajectory`, `--traj-steps`, `--load-mode`
  - batch rows may include sixth column `linear|ortho|reverse`
  - output keeps legacy keys `disp_traj`, `traj_fracs`, `load_mode`,
    `load_mode_names`
- Updated `src/novbts/groundtruth/aggregate_uipc_replicates.py` to average
  `disp_traj` across K UIPC reps and preserve load-mode metadata.
- Moved `novbts.operator.loading_history` default output to `runs/phase7`.
- Added `infra/gen_uipc_trajectory_phase7.sh`:
  - endpoint-controlled UIPC trajectory GT
  - same endpoint repeated under `linear/ortho/reverse`
  - aggregate to `data/uipc/trajectory_phase7/shear_res24_traj_REALISTIC.npz`
  - run loading-history benchmark.
- Added handoff/spec: `codex/TASK_phase7_uipc_trajectory_history.md`.
- Validation so far: Python compile OK and synthetic aggregator test confirms
  `disp_traj` merge shape/metadata.
- Started `infra/gen_uipc_trajectory_phase7.sh 12 3 9`, then stopped it by
  request after clarifying the historical trajectory pipeline is not useful for
  the current evidence path. Partial output only:
  `data/uipc/trajectory_phase7/sweep` has `42/108` replicate files. Do not use
  this partial trajectory dataset as paper evidence.
- Later resumed the same Phase 7 trajectory job on request. At commit time it is
  still an uncommitted/untracked run artifact, not part of the paper evidence.

## Previous checkpoint (2026-06-30 00:20 +07)

Realistic geometry reground da pass end-to-end acceptance.

Post-review correction (2026-06-30 00:27 +07):

- Fixed stale old-run headline numbers that remained in `docs/kse2026/main.tex`
  abstract, contribution list, and RQ4 text:
  - FNO vs MLP now `12.14x`
  - classical/analytic VBTS baseline range now `5.54--12.63x`
  - control target-query advantage now `84x`
  - control wall speedup now `24.5x`
  - RQ4 losses/walls/oracle now match `runs/phase4/policy_servo.json`
- Rephrased the broad "whole sensor differentiable end-to-end" language to an
  autograd-connected image-space loop, consistent with the current env
  finite-difference diagnostic.
- Extended `infra/verify_realistic_reground.py` to fail on these stale headline
  numbers and require the refreshed ones.
- Rebuilt `docs/kse2026/main.pdf`; verifier passed
  `REALISTIC_REGROUND_ACCEPTANCE_OK` again.

Refactor/cleanup pass:

- Replaced remaining "end-to-end differentiable sensor" wording in
  `docs/kse2026/main.tex` with "autograd-connected sensor pipeline" wording.
- Updated `docs/kse2026/README.md` so the submission checklist no longer cites
  stale one-frame inversion numbers; it now points to
  `runs/phase5/sensor_inverse_multiframe.json`.
- Removed transient root progress files (`fem_progress*.txt`, `convergence.txt`)
  and generated LaTeX intermediates (`main.aux`, `main.blg`, `main.out`).
- Removed source/script `__pycache__` directories; left virtualenv caches alone.
- Kept datasets, runs, logs, `*.PRE_REALISTIC.*` backups, `main.pdf`, and
  `main.log` because they are provenance or verifier inputs.
- Rebuilt paper and reran verifier; acceptance still passes.

Runtime/state:

- `uipc-sweep-realistic-main.service`: `inactive` / unit not found; khong restart
- Docker containers: `0`
- downstream runner: exited `REALISTIC_DOWNSTREAM_DONE`
- final verifier: `REALISTIC_REGROUND_ACCEPTANCE_OK`
- legacy dataset unchanged:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Final dataset:

- `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- N=`2520`, train/test=`2120/400`
- modes all normal/stick/partial/full = `425/712/895/488`
- test modes = `68/113/142/77`
- K=`3`, gel_res=`24`, bilinear marker sampling
- mean K3 solve time `31.843 s`; fair single-solve `10.614 s`
- mean non-normal tangential replicate noise `2.378%`
- split metadata present: `split_test_size=400`, `split_shuffle_seed=2026`

Fresh downstream outputs:

- `runs/phase3_fem/benchmark.json`
  - FNO rel-L2 overall `0.041`, direction `1.6 deg`
  - MLP rel-L2 overall `0.501`, direction `83.9 deg`
  - FNO/MLP rel-L2 advantage `12.14x`
  - RQ3 FNO `7839 fps`, fair GT solver `0.094 fps`, K3 target `0.031 fps`
  - speedup vs fair single solve `83204x`
- `runs/phase3_fem/vbts_baselines.json`
  - FNO advantages:
    - TACTO-style `12.63x`
    - Cattaneo--Mindlin `11.89x`
    - Taxim/FOTS linear `5.54x`
    - DeepONet `1.12x`
    - U-Net `1.42x`
    - Galerkin Transformer `2.19x`
- `runs/phase4/policy_servo.json`
  - autograd final `5.70e-10`, ES final `6.21e-10`
  - autograd reaches target in `21` fwd queries vs ES `1771` (`84x` fewer)
  - wall `13 s` vs `324 s`
- `runs/phase5/sensor_build.json`
  - cosine by mode normal/stick/partial/full =
    `-0.396/0.559/0.918/0.967`
  - round-trip px by mode normal/stick/partial/full =
    `0.232/0.394/0.868/1.500`
- `runs/phase5/sensor_inverse_multiframe.json`
  - overall magnitude error `15.51%`, direction `3.79 deg`
  - by-mode magnitude normal/stick/partial/full =
    `30.09%/7.35%/11.02%/13.60%`
- `runs/phase6/env_demo.json`
  - policy closes `99.93%` random-to-oracle reward gap
  - gradients flow to action
  - finite-difference diagnostic rel error `5.21%`, `passed=false`; paper now
    phrases this honestly as a sanity check, not a formal pass

Figures regenerated and accepted:

- `docs/kse2026/figs/fidelity_speed.png`
- `docs/kse2026/figs/policy_servo_curve.png`
- `docs/kse2026/figs/sensor_gt_vs_fno.png`

Paper:

- `docs/kse2026/main.tex` updated for realistic `20 x 20 x 3 mm` geometry,
  new counts, downstream metrics, solver speed, no stale flat-punch/OOD claim
- rebuilt `docs/kse2026/main.pdf` with
  `pdflatex -> bibtex -> pdflatex -> pdflatex`
- verifier confirmed no undefined refs/citations and semantic stale-claim gate
  passed

Acceptance command that passed:

```bash
rtk proxy .venv-gate2/bin/python infra/verify_realistic_reground.py
```

## Latest checkpoint (2026-06-29 17:20 +07)

Production GT van dang chay va la uu tien chinh:

- systemd user unit: `uipc-sweep-realistic-main.service`
- status: `active`
- current container: `uipcsweep_45`
- complete combos: `46 / 63`
- averaged frames: `1840 / 2520` (`73.02%`)
- raw replicates implied/present: `5520 / 7560` (`73.02%`)
- final realistic NPZ:
  `data/uipc/shear_res24_avg_swept_REALISTIC.npz` chua ton tai, dung nhu
  expected truoc khi full sweep xong

Pre-Phase-2 preparation da lam trong luc cho GT:

- `py_compile` pass:
  - `infra/verify_realistic_reground.py`
  - `src/novbts/groundtruth/aggregate_uipc_replicates.py`
- `bash -n` pass:
  - `infra/run_realistic_downstream.sh`
  - `infra/gen_uipc_sweep.sh`
- `infra/verify_realistic_reground.py` da chay thu va fail dung tai:
  missing `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- verifier xac nhan legacy dataset van unchanged:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Khong nen launch them rehearsal/training nang. Cho production finish, sau do lam
theo thu tu:

1. Aggregate/gate final NPZ. Neu final NPZ thieu split metadata, rerun host
   aggregator voi `--test-size 400 --shuffle-seed 2026`.
2. Chay `infra/verify_realistic_reground.py`; dataset gate phai pass truoc Phase
   2.
3. Chay `infra/run_realistic_downstream.sh` voi realistic NPZ.
4. Cap nhat paper va rebuild sau khi co final downstream metrics.

## CPU partial-data rehearsal (2026-06-29)

Theo yeu cau tiep tuc thu downstream trong khi GPU production chay, da tao
snapshot rieng, khong overwrite active/final artifacts:

- snapshot: `data/uipc/rehearsal/shear_realistic_partial.npz`
- N=`964`, split stratified `764/200`, seed `2026`
- modes all=`30/310/402/222`
- modes test=`6/64/84/46`
- K=3, finite, non-normal tangential noise `2.178%`

Da them path overrides voi defaults khong doi trong `src/novbts/paths.py`:

- `NOVBTS_RUNS_DIR`
- `NOVBTS_FEM_DIR`
- `NOVBTS_DOCS_DIR`

Rehearsal chay CPU-only (`CUDA_VISIBLE_DEVICES=''`, nice priority, 4 threads),
epochs/steps rut gon de gate entrypoint/schema, khong dung metrics nay cho paper:

- benchmark -> `runs/rehearsal_realistic_partial/phase3_fem/benchmark.json`
- VBTS bake-off -> `runs/rehearsal_realistic_partial/phase3_fem/vbts_baselines.json`
- policy -> `runs/rehearsal_realistic_partial/phase4/policy_servo.json`
- sensor build -> `runs/rehearsal_realistic_partial/phase5/sensor_build.json`
- inversion -> `runs/rehearsal_realistic_partial/phase5/sensor_inverse_multiframe.json`
- env/gradcheck -> `runs/rehearsal_realistic_partial/phase6/env_demo.json`
- 3 figures -> `docs/rehearsal_realistic_partial/kse2026/figs/`

Ket qua pipeline:

- tat ca entrypoints exit 0
- env finite-difference gradcheck pass, rel error `1.49%`
- sensor JSON co cosine/round-trip per mode
- RQ3 JSON tach single solve `0.093 fps` va K3 target `0.031 fps`
- 9 rehearsal JSON co dung `gt`/`gt_path`

Rehearsal phat hien va da sua:

- concurrent snapshot aggregator khong duoc ghi avg vao root-owned active combo:
  them `--no-write-frame-averages`
- `sensor_inverse.json` bo sot `gt`/`gt_path` trong nested output: da them va
  rerun verify

Luu y: model final van phai retrain tren full N=2520. Snapshot rehearsal chi
chung minh commands/schema/output flow san sang; khong the chi append GT vao
weights rehearsal va coi la final scientific result.

### Full 80-epoch partial rehearsal (stopped de uu tien production)

Theo yeu cau user, da launch full-settings rehearsal tren cung frozen N=964
snapshot:

- initial 4-thread unit `realistic-partial-80ep.service` da stop sau 8 phut theo
  yeu cau user; no van o benchmark va chua sinh benchmark JSON
- current systemd unit: `realistic-partial-80ep-12t.service`
- CPU-only: `CUDA_VISIBLE_DEVICES=''`
- `OMP_NUM_THREADS=12`, `MKL_NUM_THREADS=12`, nice level `5`
- script: `infra/run_realistic_partial_80ep.sh`
- log: `logs/realistic_partial_80ep_12t.log`
- runs: `runs/rehearsal_realistic_partial_80ep/`
- FEM/sensor data: `data/fem/rehearsal_realistic_partial_80ep/`
- figures: `docs/rehearsal_realistic_partial_80ep/kse2026/figs/`

Sequence dung full settings:

- benchmark 80 epochs + classifier 40
- VBTS bake-off 80 epochs
- policy FNO 80 epochs + 300-step, 3-seed autograd/ES
- sensor build
- sensor inversion FNO 80 epochs + 400 steps, 5 frames/mode, 8 restarts
- env FNO 80 epochs + 300-step policy + gradcheck
- 3 figures

12-thread relaunch gate pass: N=964, train/test=764/200. Python dang dung
~`1147%` CPU (xap xi 12 cores) o `fem_benchmark`. Production service
`uipc-sweep-realistic-main` van active, khong restart.

Status full-80 rehearsal luc `2026-06-29 09:52 +07`:

- `fem_benchmark` da xong, JSON da ghi
- partial-snapshot RQ1 (rehearsal, khong phai final paper):
  - MLP overall `0.482`, direction `88.1 deg`
  - FNO overall `0.056`, direction `2.4 deg`
  - FNO/MLP rel-L2 advantage `8.55x`
  - FNO per-mode normal/stick/partial/full =
    `0.031/0.049/0.058/0.068`
  - FNO throughput `625 fps`
  - GT single solve `0.093 fps`, speedup `6729x`
- current step: `vbts_baselines` 80 epochs
- unit `realistic-partial-80ep-12t.service` active

Status luc `11:39 +07`:

- benchmark va VBTS baselines da xong
- current step: `diff_policy` / policy servo full settings
- policy command dung 300 steps, ES population 32, 3 seeds; day la CPU-heavy
  stage va chua co `policy_servo.json`
- unit active, process khong bi treo
- 12-thread CPU load lam production solve time tang tu `~10.7 s` len
  `~16--17 s`; GT van tien trien nhung cham hon

**Stopped theo yeu cau user:** luc sau `11:39 +07`, user quyet dinh rehearsal
da du va uu tien production GT. Da stop rieng
`realistic-partial-80ep-12t.service`:

- rehearsal unit: `inactive`
- production `uipc-sweep-realistic-main`: van `active`, khong restart
- benchmark + VBTS baseline JSON duoc giu
- policy servo dang chay bi dung, khong co `policy_servo.json`
- sensor/inversion/env full-80 chua chay
- reduced smoke rehearsal truoc do van day du cho tat ca entrypoints
- production solve time sau khi tra CPU ve khoang `12--12.5 s`

## Production status moi nhat (2026-06-29)

Kiem tra tu account/session moi:

- `uipc-sweep-realistic-main.service`: `active`
- khong restart service
- current container: combo `025`
- replicate NPZ: `4568 / 7560` (`60.4%`) at `2026-06-29 14:43 +07`
- completed averaged frames: `1520 / 2520` (`60.3%`, combos `000..037`)
- combo 38 dang chay (`8/120`)
- final realistic NPZ chua ton tai, dung nhu expected truoc khi full sweep xong

Audit tat ca 920 averaged frames:

- bad/corrupt/schema mismatch: `0`
- current mode counts: normal/stick/partial/full = `27/297/384/212`
- mean K=3 target solve time: `32.066 s`
- mean single solve time: `10.689 s`
- mean non-normal tangential replicate noise: `2.201%`
- p95: `8.842%`; co outlier cao nhung mean van duoi acceptance gate `3%`
- ETA theo measured throughput: khoang `14--15 h` con lai

## Phase 1 dang chay (2026-06-29 00:03 +07)

User da tiep tuc goal sau checkpoint Phase 0. Production campaign da launch:

- systemd user unit: `uipc-sweep-realistic.service`
- main PID sau relaunch: `2593317` (PID co the doi neu restart; unit name moi la
  identifier on dinh)
- sweep root: `data/uipc/sweep_realistic`
- target: `63 combos x 40 frames = 2520 frames`
- replicates: `K=3`
- final target: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- log: `logs/uipc_sweep_realistic.log`
- progress file: `fem_progress_uipc_0_62.txt`

Campaign dung transient user systemd service thay vi `nohup`. Lan launch `nohup`
dau tien lam wrapper bash mat trong khi container combo 0 bi orphan, nen campaign
se khong chuyen combo. Container orphan da duoc stop va campaign duoc relaunch
bang systemd; sweep resumable nen rep da xong khong bi tinh lai.

Theo doi:

```bash
rtk proxy systemctl --user status uipc-sweep-realistic --no-pager
rtk docker ps
rtk proxy tail -50 fem_progress_uipc_0_62.txt
rtk proxy tail -50 logs/uipc_sweep_realistic.log
```

Neu service bi fail:

```bash
rtk proxy systemctl --user reset-failed uipc-sweep-realistic
```

Sau do relaunch cung command/cau hinh; script se skip moi rep NPZ da ton tai va
hop le. Khong chay them UIPC workload GPU song song voi service nay.

Ngay sau launch, audit rows phat hien 63 combos ban dau deu la random, trai voi
yeu cau normal-contact top-up. Campaign da duoc dung khi moi co 15 rep, cac
artifact sai da bi xoa, va sampler da duoc sua thanh campaign design:

- combos `000..049`: random
- combos `050..058`: pure normal, grid `R={2,4,6} mm x E={0.5,1,2}e5 Pa`
- combos `059..062`: extra random

Class counts du kien truc tiep tu rows (va se duoc verify lai tren final NPZ):

- normal: `425`
- stick: `712`
- partial: `895`
- full: `488`
- total: `2520`

Service sach da relaunch luc `2026-06-29 00:06:37 +07`.

Luc `00:12 +07`, campaign duoc chuyen sang hai shard GPU song song theo chinh
pattern duoc document trong `gen_uipc_sweep.sh` (~1.5x throughput):

- `uipc-sweep-realistic-a.service`: combos `000..031`
  - progress `fem_progress_uipc_0_31.txt`
  - log `logs/uipc_sweep_realistic_a.log`
  - temporary aggregate `data/uipc/shear_res24_avg_swept_REALISTIC_shard_a.npz`
- `uipc-sweep-realistic-b.service`: combos `032..062`
  - progress `fem_progress_uipc_32_62.txt`
  - log `logs/uipc_sweep_realistic_b.log`
  - temporary aggregate `data/uipc/shear_res24_avg_swept_REALISTIC_shard_b.npz`

Hai shard chung resumable sweep root, khong overlap combo/container name. Khong
shard nao ghi vao final filename. Sau khi ca hai complete, phai aggregate lai
toan bo sweep root vao dung final path roi moi chay Phase 2 gate.

**Superseded luc 00:15 +07:** thu nghiem hai shard cho thay moi solve tang tu
`~11 s` len `~60 s` voi contact config `eps=2.5e-5` (GPU contention), nen tong
throughput cham gan 3x. Ca hai shard da duoc stop, orphan container da remove.
Campaign hien tai quay lai mot service:

- `uipc-sweep-realistic-main.service`
- log: `logs/uipc_sweep_realistic_main.log`
- progress: `fem_progress_uipc_0_62.txt`
- output final truc tiep:
  `data/uipc/shear_res24_avg_swept_REALISTIC.npz`

Sau khi quay lai mot container, solve time da tro ve `11.2--11.3 s`. Mot rep cua
combo 32 tu shard test duoc giu; no cung config va se duoc resumable batch skip
khi main service toi combo 32.

User systemd ban dau co `Linger=no`, nen service co nguy co dung neu tat het
session khi chuyen account. Da chay `loginctl enable-linger tungmai`; hien
`Linger=yes`, service se tiep tuc qua full logout/relogin.

Phase 2 da duoc chuan bi trong:

- `infra/run_realistic_downstream.sh`

Script nay:

- gate final NPZ: exactly 2520 frames, modes 0..3, K=3, res24, chosen contact
  params va bilinear marker sampling
- backup artifact cu thanh `*.PRE_REALISTIC.*` mot lan truoc khi overwrite
- chay benchmark, VBTS baselines, policy servo, sensor build, sensor inversion,
  env demo/gradcheck, figures theo thu tu dependency
- copy fresh `runs/phase5/gt_vs_fno_samples.png` vao paper figures

Tat ca JSON output Phase 2 da duoc bo sung provenance top-level:

- `gt = shear_res24_avg_swept_REALISTIC.npz`
- `gt_path = data/uipc/shear_res24_avg_swept_REALISTIC.npz`

Final UIPC aggregator cung da duoc sua de concatenate `velocity_tol`; schema nay
da test bang Phase-0 K=3 smoke artifact.

Acceptance verifier da duoc them:

- `infra/verify_realistic_reground.py`

Verifier gate dataset/schema/parameter box, JSON freshness + provenance, figures,
stale paper claims va LaTeX undefined refs. Hien tai no fail dung o missing final
NPZ vi Phase 1 chua xong.

Hai schema/metric gap duoc bat khi audit truoc Phase 2:

- `sensor_build.json` bay gio ghi `flow_disp_cos_by_mode` va
  `round_trip.by_mode_px` cho normal/stick/partial/full, thay vi chi blanket
  cosine + stick/slip.
- RQ3 benchmark bay gio tach:
  - `gt_solver`: fair single-solve FPS = `solve_time_s / n_replicates`
  - `gt_solver_k3_averaged`: production target FPS tinh ca K=3 calls
  Figure dung single-solve line; JSON note noi ro chi phi K=3.

Verifier da gate ca hai invariant nay va gate FNO rel-L2 < MLP rel-L2.

Split audit phat hien dataset IPC cu duoc aggregate theo combo order, lam last-400
test set bi top-up skew (`247 normal`, chi `34 full`). Final realistic aggregator
da duoc them deterministic stratified train/test ordering:

- `split_test_size=400`
- `split_shuffle_seed=2026`
- train roi test van nam theo convention first 2120 / last 400
- voi row counts du kien, test modes se xap xi `68/113/142/77`
- `source_frame_dir` duoc luu de trace moi shuffled frame ve artifact goc

Do main service da launch truoc edit nay, running bash co the van giu old script
inode. Sau Phase 1 phai gate split metadata; neu thieu, rerun host aggregator:

```bash
rtk proxy .venv-gate2/bin/python -m novbts.groundtruth.aggregate_uipc_replicates \
  --sweep-dir data/uipc/sweep_realistic \
  --out data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  --mode-shear-scale 0.001 --expect-reps 3 \
  --test-size 400 --shuffle-seed 2026
```

Downstream runner va final verifier deu tu choi dataset neu split metadata/test
mode coverage khong dung.

Early production integrity check (combo 0, 20 complete frames):

- rows/depth/shear/mode label match: `62/62` rep artifacts checked, `0` errors
- no K replicate pair bitwise-identical
- mean raw tangential replicate noise:
  - stick `1.81%`
  - partial `2.71%`
  - full `2.87%`
- co mot so outlier `~7--8%`, nen final verifier gate mean non-normal
  tangential replicate noise `<=3%` truoc khi Phase 2 duoc phep chay.

Host-side K=3 aggregation da duoc test tren production frame 000:

- `solve_time_s=35.418 s` bang tong ba solve
- mean single solve `11.806 s`
- tangential replicate noise `1.862%`
- schema/provenance va shapes deu dung

Legacy dataset no-overwrite invariant da duoc khoa trong final verifier:

- path: `data/uipc/shear_res24_avg_swept.npz`
- size: `15,296,208 bytes`
- SHA-256:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Verifier fail neu legacy file bi sua/xoa.

Acceptance grep scope note:

- raw grep toan bo `src/runs` khong the zero-match hop ly:
  - analytic baseline source co support `flat-punch` that su
  - archived/PRE_TOPUP JSON co old numbers va phai duoc giu
  - `0.975`/`0.341` cung co the xuat hien nhu mot metric khac do trung chuoi
- final verifier gate semantic target:
  - `docs/kse2026/main.tex` khong con stale claims/numbers
  - 6 active downstream JSON phai moi hon realistic NPZ va co exact realistic
    `gt`/`gt_path`
  - figures phai moi hon JSON dependency
- archived/PRE artifacts khong duoc dung lam evidence, nhung khong xoa vi task
  yeu cau backup.

Static traceability fix da ap dung trong `docs/kse2026/README.md`:

- sensor cosine/round-trip -> `runs/phase5/sensor_build.json`
- inversion by regime -> `runs/phase5/sensor_inverse_multiframe.json`
- env source giu `runs/phase6/env_demo.json` nhung bo stale hard-coded 87%

Physical BC da duoc audit truc tiep trong
`src/novbts/groundtruth/tacex_uipc_extract_shear.py`:

- structured gel z spans `0..gel_z`
- chi bottom mask (`z=min`) duoc set `is_constrained=1` va aim ve rest position
- top va sides khong co constraint
- toan bo rigid/stiff indenter duoc prescribed theo displacement schedule
- UIPC ground half-plane dat tai `-gel_z`, khong support gel

Vi vay paper wording "bottom bonded/pinned Dirichlet, top contact surface, free
sides, rigid displacement-controlled indenter" dung voi implementation. Khong
them provenance key moi vao driver khi production dang chay, de tranh combo 0 va
cac combo sau khac schema.

## Cap nhat checkpoint Phase 0 (2026-06-28 17:01 +07)

Phase 0 da hoan tat. Cau hinh duoc chon:

- geometry: `20 x 20 x 3 mm`
- structured tet, `gel_res=24`, marker grid `32 x 32`
- marker sampling: `bilinear`
- `eps_velocity=0.000025`
- `d_hat=0.0001`
- `contact_resistance=1.0e9`
- `velocity_tol=0.001`
- production depth: `U(0.15, 0.75) mm`
- production radius: `U(2, 6) mm`
- production shear: `g * mu * 0.001 m`, `g in [0, 1.3]`
- `K=3`

Eps-axis tai res24:

- `0.001 -> 0.0005`: tangential `29.21%`
- `0.0005 -> 0.00025`: tangential `19.14%`
- `0.00025 -> 0.0001`: tangential `10.27%`
- `0.0001 -> 0.00005`: tangential `4.97%`
- `0.00005 -> 0.000025`: tangential `1.54%`, normal `0.42%`

Report:
`data/uipc/conv_realistic_eps_axis_vtol001/convergence_report.json`.

Do do, lua chon cu `eps_velocity=0.00025` da bi loai. Default trong hai
infra scripts da duoc doi sang contact config moi.

Production-shaped smoke:

- artifact:
  `data/uipc/sweep_realistic_phase0_ev000025/combo_000/frame_000/uipc_gt_shear_avg.npz`
- one realistic random frame, depth `0.614 mm`, mode partial
- `K=3`
- replicate noise: tangential `1.02%`, normal `0.10%`
- mean single-solve time `11.79 s`; K=3 sum `35.38 s`
- provenance co `gel_res`, `eps_velocity`, `velocity_tol`, `d_hat`,
  `contact_resistance`, `marker_sampling`, `n_replicates`

Regime check tai chinh chosen config:

- artifact root: `data/uipc/realistic_phase0/chosen_eps_regimes`
- normal: mode 0, shear `0 mm`
- stick: mode 1, shear `0.12 mm`
- partial: mode 2, shear `0.42 mm`
- full: mode 3, shear `0.69 mm`

Mesh gate truoc do van la:

- K=3 averaged res24 -> res28: tangential `0.95%`, normal `1.18%`
- K=3 res24 replicate noise: tangential `1.56%`
- K=3 res28 replicate noise: tangential `2.18%`
- res20 -> res24 khong sach (`~12%`), nen res20 qua coarse cho thin gel;
  res24 -> res28 moi la plateau co y nghia.

`infra/gen_uipc_sweep.sh` bay gio co `OUT_DATA` override de Phase-0 smoke
khong ghi nham vao final production filename. File final 1-frame tao tam trong
smoke da bi xoa; `data/uipc/shear_res24_avg_swept_REALISTIC.npz` hien khong ton
tai, dung nhu trang thai truoc Phase 1.

Checkpoint tiep theo: user review Phase 0 truoc khi launch detached Phase 1.
Lenh production de xuat:

```bash
rtk proxy nohup env \
  SWEEP_DIR=data/uipc/sweep_realistic \
  OUT_DATA=data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  GEL_RES=24 EPS_VELOCITY=0.000025 D_HAT=0.0001 \
  CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 63 40 3 \
  > logs/uipc_sweep_realistic.log 2>&1 &
```

## Goal ban dau

Doc va thuc hien `codex/TASK_realistic_geometry_reground.md`.

Muc tieu lon:
- Regenerate IPC/UIPC ground truth tren hinh hoc gel realistic cho VBTS:
  - gel footprint `20 x 20 mm`
  - gel thickness `3 mm`
  - representative, sensor-agnostic, khong gan voi DIGIT/GelSight cu the
- Phase 0 phai validate stability truoc khi chay full production sweep.
- Phase 1 tao final npz moi:
  - `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - khong overwrite `data/uipc/shear_res24_avg_swept.npz`
- Phase 2 rerun all downstream voi npz realistic.
- Phase 3 update paper `docs/kse2026/main.tex` va rebuild.
- User yeu cau nen dung subagents de lam song song; da dung subagents cho codebase inspection va downstream patch plan.

## File da sua

Working tree hien tai co cac file modified:

- `infra/gen_uipc_convergence.sh`
  - Chuyen default convergence sang realistic geometry.
  - Them env overrides: `GEL_XY`, `GEL_Z`, `INDENTOR_R`, `MU`, `YOUNGS`, `DEPTH`, `SHEAR`, `DRIVE_RATIO`, `D_HAT`, `CONTACT_RESISTANCE`, `VELOCITY_TOL`, `RES_LEVELS_STR`, `EPS_LEVELS_STR`, `CONV_DIR`.
  - Them `chown` sau container run de tranh root-owned output.

- `infra/gen_uipc_sweep.sh`
  - Chuyen sweep output sang `data/uipc/sweep_realistic`.
  - Scale realistic:
    - depth `U(0.00015, 0.00075)`
    - radius `U(0.002, 0.006)`
    - shear scale `g * mu * 0.001`
  - Them final aggregate ra `data/uipc/shear_res24_avg_swept_REALISTIC.npz`.
  - Them env overrides: `GEL_RES`, `EPS_VELOCITY`, `D_HAT`, `CONTACT_RESISTANCE`, `VELOCITY_TOL`, `GEL_XY`, `GEL_Z`, `SWEEP_DIR`.

- `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - Doi marker sampling tren top face tu nearest-neighbor KDTree sang bilinear interpolation tren structured top grid.
  - Neu top grid khong rectangular thi fallback ve nearest.
  - Luu provenance `marker_sampling` vao npz.

- `src/novbts/groundtruth/aggregate_uipc_replicates.py`
  - So sanh string-safe trong `_same`.
  - Check/copy `marker_sampling` qua averaged frames va final sweep aggregate.

- `src/novbts/groundtruth/aggregate_uipc_convergence.py`
  - Assert cac convergence fields paired dung:
    - `coords`
    - `params`
    - `mode`
    - `d_hat`
    - `contact_resistance`
    - `marker_sampling`
  - Luu `marker_sampling` vao convergence report.

- `src/novbts/operator/fem_benchmark.py`
  - Load GT provenance tu selected `--data` npz.
  - Solver FPS lay tu `solve_time_s` trong input npz, khong scan stale FEM/PhysX files nua.
  - Them `gt_path`, `gt_provenance`, data-derived `param_box`.
  - Them generic `gt_solver`, giu alias `physx_fem_shear_solver` de compat.

- `src/novbts/report/make_kse_figs.py`
  - Figure generation prefer `throughput_fps["gt_solver"]`, fallback alias cu.

## Phase 0 artifacts va ket qua hien tai

Artifacts chinh da tao:

- Existing/single checks:
  - `data/uipc/realistic_phase0/smoke/uipc_gt_shear.npz`
  - `data/uipc/realistic_phase0/singles/{normal,stick,partial,full}/uipc_gt_shear.npz`

- Convergence/probe dirs:
  - `data/uipc/conv_realistic/`
  - `data/uipc/conv_realistic_ev0005/`
  - `data/uipc/conv_realistic_ev0005_dhat0002/`
  - `data/uipc/conv_realistic_ev00025/`
  - `data/uipc/conv_realistic_ev0001/`
  - `data/uipc/conv_realistic_bilinear_ev00025/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030_res24_28/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030_vtol001/`

- K-replicate checks:
  - `data/uipc/realistic_phase0/krep_ev0005/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_res24_28/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res20/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res24/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res28/`

Important numbers:

- 4 regime single checks at realistic geometry passed and produced valid npz:
  - normal mode 0
  - stick mode 1
  - partial mode 2
  - full mode 3

- Earlier default-ish settings failed convergence:
  - `eps=0.001`, `d_hat=0.0001`, `depth=0.45mm`: res20->24 tangential rel-L2 about `42%`.
  - `eps=0.0005`: improved but still about `11.5%`.

- Bilinear marker sampling alone did not fix convergence:
  - `eps=0.00025`, `depth=0.45mm`: res20->24 tangential about `11.4%`.

- Shallower validation point helped:
  - `depth=0.30mm`, `eps=0.00025`, bilinear: res20->24 tangential about `3.77%`, normal about `11.6%`.

- Tight velocity tolerance was the key for repeatability:
  - `depth=0.30mm`, `eps=0.00025`, `d_hat=0.0001`, `velocity_tol=0.001`, bilinear.
  - K=3 res24:
    - tangential rep noise `1.56%`
    - normal rep noise `0.42%`
  - K=3 res28:
    - tangential rep noise `2.18%`
    - normal rep noise `0.43%`
  - K=3 averaged res24->28:
    - overall `1.14%`
    - normal `1.18%`
    - tangential `0.95%`
  - K=3 averaged res20->24:
    - overall `11.70%`
    - normal `11.63%`
    - tangential `12.05%`

Interpretation hien tai:
- `res20` qua coarse cho thin-layer geometry.
- `res24` va `res28` co ve da vao plateau khi dung K=3 + `velocity_tol=0.001`.
- Phase 0 co dau hieu tot o `res24->28`, nhung chua xong eps-axis test.

## Loi/test hien tai

Verification da chay thanh cong:
- `py_compile` cho cac Python files modified:
  - `tacex_uipc_extract_shear.py`
  - `aggregate_uipc_replicates.py`
  - `aggregate_uipc_convergence.py`
  - `fem_benchmark.py`
  - `make_kse_figs.py`
- `bash -n` cho:
  - `infra/gen_uipc_convergence.sh`
  - `infra/gen_uipc_sweep.sh`
- `rtk docker ps` sau interrupt gan day: khong co container dang chay.

Known issue / dang dang do:
- Eps-axis convergence run vua bi user interrupt, chua tao artifact:
  - intended dir: `data/uipc/conv_realistic_eps_axis_vtol001`
  - `rtk find data/uipc/conv_realistic_eps_axis_vtol001` luc do tra `0`.
- Chua chay production sweep Phase 1.
- Chua tao final realistic npz.
- Chua rerun downstream Phase 2.
- Chua update/rebuild paper Phase 3.

Possible gotcha:
- Some shell one-liners truoc do bi quote issue khi print Python f-string, nhung data aggregate van OK.
- Container writes root-owned files; scripts now chown, but manual docker probes phai nho chown.

## Viec con lai

Immediate next steps:

1. Hoan tat eps-axis convergence tai res24 voi chosen Phase-0 knobs:

```bash
rtk proxy env CONV_DIR=data/uipc/conv_realistic_eps_axis_vtol001 \
  GEL_XY=0.020 GEL_Z=0.003 INDENTOR_R=0.004 MU=0.6 YOUNGS=1.0e5 \
  DEPTH=0.00030 SHEAR=0.00036 DRIVE_RATIO=0.60 \
  D_HAT=0.0001 CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.001 \
  RES_LEVELS_STR="24" EPS_LEVELS_STR="0.001 0.0005 0.00025 0.0001" \
  bash infra/gen_uipc_convergence.sh

rtk proxy .venv-gate2/bin/python -m novbts.groundtruth.aggregate_uipc_convergence \
  --conv-dir data/uipc/conv_realistic_eps_axis_vtol001
```

2. Neu eps-axis sach, chot Phase 0 params:
   - `gel_res=24`
   - `gel_xy=0.020`
   - `gel_z=0.003`
   - bilinear marker sampling
   - `eps_velocity=0.00025`
   - `d_hat=0.0001`
   - `velocity_tol=0.001`
   - `contact_resistance=1.0e9`
   - consider production depth range:
     - original task says `0.15-0.75mm`
     - Phase 0 validation is cleanest at `0.30mm`; 0.45mm showed mesh drift at res20->24, but res24->28/K=3 with tight tol was good at 0.30mm.
     - Need decide whether to keep full original depth range or narrow upper depth before full sweep.

3. Optional but useful before full production:
   - production smoke:

```bash
rtk proxy env SWEEP_DIR=data/uipc/sweep_realistic_phase0 \
  GEL_RES=24 EPS_VELOCITY=0.00025 D_HAT=0.0001 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 1 1 3
```

4. When Phase 0 accepted, run Phase 1 detached with `nohup`, not Codex background:

```bash
rtk proxy nohup env \
  SWEEP_DIR=data/uipc/sweep_realistic \
  GEL_RES=24 EPS_VELOCITY=0.00025 D_HAT=0.0001 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 63 40 3 \
  > logs/uipc_sweep_realistic.log 2>&1 &
```

Expected final output:
- `data/uipc/shear_res24_avg_swept_REALISTIC.npz`

5. After final npz exists:
   - load and count frames/classes
   - verify all 4 regimes populated
   - verify K=3 averaged
   - verify solve-time provenance

6. Phase 2 downstream reruns with:
   - `--data data/uipc/shear_res24_avg_swept_REALISTIC.npz`
   - backup old JSONs as `*.PRE_REALISTIC.json` before overwrite.

7. Phase 3 paper update:
   - update `docs/kse2026/main.tex`
   - remove stale claims/numbers:
     - `0.975`
     - `0.341`
     - `50\times50`
     - `490/430`
     - `flat-punch`
   - rebuild `pdflatex -> bibtex -> pdflatex -> pdflatex`.

## Tieu chi hoan thanh

Goal chi complete khi tat ca dung:

- Phase 0:
  - realistic geometry confirmed:
    - `20 x 20 x 3 mm`
    - structured tet box only
  - smoke/single/convergence clean enough.
  - chosen params documented:
    - `gel_res`
    - `eps_velocity`
    - `d_hat`
    - `velocity_tol`
    - `contact_resistance`
    - depth/shear scales
  - all 4 regimes populated in checks.

- Phase 1:
  - final npz exists and loads:
    - `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - N about `2520`
  - all 4 regimes populated
  - K=3 averaged
  - no overwrite of old `data/uipc/shear_res24_avg_swept.npz`

- Phase 2:
  - all downstream JSONs regenerated with realistic npz:
    - `runs/phase3_fem/benchmark.json`
    - `runs/phase3_fem/vbts_baselines.json`
    - `runs/phase4/policy_servo.json`
    - `runs/phase5/sensor_build.json`
    - `runs/phase5/sensor_inverse_multiframe.json`
    - `runs/phase6/env_demo.json`
  - provenance points to realistic npz.
  - figures regenerated:
    - `docs/kse2026/figs/fidelity_speed.png`
    - `docs/kse2026/figs/policy_servo_curve.png`
    - `docs/kse2026/figs/sensor_gt_vs_fno.png`

- Phase 3:
  - paper uses real new geometry `20 x 20 x 3 mm`.
  - false `1.3%` validator claim removed/reframed.
  - speedup uses real IPC `solve_time_s` from realistic npz.
  - flat-punch OOD claim removed unless a real flat-punch dataset is generated.
  - RQ5 reports per-regime cosine/round-trip on realistic data.
  - limitations updated to sim-to-real/no physical sensor, not old too-large gel.
  - build has 0 undefined refs.

- Final acceptance grep clean:

```bash
rtk grep "0.975|0.341|50\\\\times50|490/430|flat-punch" docs/kse2026 src runs
```

## 2026-06-30 report reground cleanup

- Rewrote current phase/report markdowns to use only realistic IPC/UIPC metrics:
  - `docs/bao_cao_giai_doan3_rq_results.md`
  - `docs/bao_cao_tong_ket_phase3-6.md`
  - `docs/bao_cao_phase7_gipc_convergence.md`
  - `docs/kse2026/referee_report.md`
- Removed generated stale report artifacts from `docs/`:
  - `docs/bao_cao_tong_ket_phase3-6.pptx`
  - `docs/bao_cao_tong_ket_phase3-6.pdf`
  - `docs/bao_cao_giai_doan3.pdf`
  - `docs/bao_cao_phase7_gipc_convergence.pdf`
  - `docs/bao_cao_giai_doan5_sensor.pdf`
- Removed untracked pre-realistic figure backups and rehearsal figure copies after current figures were verified.
- Current root-level report files now point to:
  - dataset `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - N=2520, K=3, gel `20 x 20 x 3 mm`
  - Phase 3 FNO `0.041 / 1.6 deg`, MLP `0.501 / 83.9 deg`, speedup `83204x`
  - Phase 4 autograd vs ES `300` vs `19200` forward evals
  - Phase 5 multi-frame inverse `15.51% / 3.79 deg`
  - Phase 6 reward gap closed `99.93%`, gradcheck diagnostic `passed=false`
- Verification:
  - stale headline grep over current reports is clean for old numeric literals.
  - `infra/verify_realistic_reground.py` passes `REALISTIC_REGROUND_ACCEPTANCE_OK`.

## 2026-06-30 runs image refresh

- Cleaned `runs/` to keep only current realistic outputs used by paper/verifier.
- Regenerated run figures from realistic outputs:
  - `runs/phase3_fem/fidelity_speed.png` from `runs/phase3_fem/benchmark.json`
  - `runs/phase4/policy_servo_curve.png` from `runs/phase4/policy_servo.json`
  - `runs/phase5/preview.png` and `runs/phase5/test_samples.png` from realistic sensor build
  - `runs/phase6/env_demo.png` from current tactile env demo
- Latest Phase 6 env demo:
  - reward gap closed `99.88%`
  - finite-difference diagnostic rel error `1.56e-4`
  - `gradcheck.passed=true`
