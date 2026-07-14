# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **Fourier Neural Operator (FNO) surrogate for Vision-Based Tactile Sensors (VBTS / GelSight)**.
The scientific contribution is the FNO: it maps contact parameters / load to the gel deformation
field ~10⁴× faster than a FEM solver, is differentiable (used for control), and beats MLP/DeepONet
and analytic (Hertz / Cattaneo–Mindlin) baselines on slip prediction. Everything else (PhysX vs IPC
ground truth, the sensor renderer) is supporting framework, not the headline.

Code lives in the installable package `novbts` (under `src/`). The repo is a git repo (default
branch `main`; current work on `phase4-diff-policy`). `README.md` at the root is the canonical map.

## Setup & running

- Python 3.10, editable install into a project venv: `.venv-gate2/bin/pip install -e .`
- All paths resolve from the repo root via `novbts.paths` (ROOT/DATA/ANALYTIC/FEM/RUNS/DOCS), so
  host modules run from any CWD. **No `sys.path` hacks in host modules** — always run as modules
  (the Docker-standalone GT/sim drivers under `research/groundtruth/isaac_*`,
  `research/groundtruth/tacex_*`, and `simulation/` are the intentional exception: they bootstrap
  `/work/src`):
  - `.venv-gate2/bin/python -m novbts.research.fno.field2field`  (headline field→field FNO)
  - `python -m novbts.research.fno.fem_train_compare` / `.eval_rq` / `.fem_benchmark`
  - `python -m novbts.research.report.make_pdf`
- There is no formal test suite / linter; "tests" here are smoke runs of the drivers (e.g. the
  `--smoke` flag on GT drivers) and the benchmark/eval modules.

## Package layout (big picture)

The package splits in two: `research/` (the science — importable, runs on the host via the editable
install) and `simulation/` (Isaac Sim drivers — standalone inside Docker, see the note below).

`src/novbts/`
- `paths.py` — single source of truth for all data/run/docs locations.
- `research/groundtruth/` — generators for ground-truth deformation fields:
  - Analytic: `hertz_mindlin`, `data_gen`.
  - PhysX FEM (Isaac Sim): `isaac_extract_normal`, `isaac_extract_shear` — **self-contained,
    run standalone inside Docker, do NOT import novbts**.
  - IPC/GIPC (libuipc/TacEx): `tacex_uipc_extract_shear.py` (current GT backend, see below) +
    `aggregate_uipc_replicates.py`, `aggregate_uipc_convergence.py`, `contact_imprint.py`
    (raycast contact profiles for mesh indentors).
- `research/fno/` — the FNO and training/eval: `field2field` (headline), `param2field`,
  `eval_rq`, `fem_train_compare`, `fem_benchmark`, `vbts_baselines` (bake-off incl. the analytic
  baselines), `geometry_ood`, `diff_policy` (control policy through the differentiable FNO, Phase 4).
- `research/models.py` — FNO / MLP / DeepONet / SpectralConv2d definitions (single module).
- `research/report/` — PDF/slide generators (`make_pdf`, `make_summary_pdf`, `make_slides`,
  `make_phase5_pdf`, `make_kse_figs`).
- `research/validation/` — GT validation/comparison utilities.
- `simulation/sensor/` — differentiable marker-dot VBTS renderer (`tactile_env.py`, `markercam.py`,
  camera + dot render). Host-importable.
- `simulation/runtime/` — Isaac Sim grasp env (UR3e + Robotiq 2F-85 + VBTS): `grasp_demo.py` +
  the `grasp_*` modules it was split into, plus `build_robot_asset.py`, `gel_contact.py`,
  `vbts_sensor.py`, `grasp_tactile_overlay.py` (screen-fixed `omni.ui` tactile HUD, never a USD
  prim), `fno_export.py`, `sensor_demo.py`. **Like the GT extractors, these Isaac drivers run
  standalone inside Docker and bootstrap `/work/src` onto `sys.path` — they do NOT rely on the
  editable install and do NOT import as `novbts.*` at runtime.** Launched via
  `infra/run_sim_grasp.sh` (not `docker` directly — bare `docker` exits 126 here). See memory
  `isaac-grasp-*` for gate state.
Dead code / PoCs live in `scripts/archive/` (NOT in the package). Infra (Dockerfiles, sweep
shell scripts) in `infra/`.

**`params` column schema** (GT npz): `[x0, y0, depth, radius, sx, sy, mu, E, geom_code, radius2]`.
It grew 9 → 10 columns (`radius2` appended for ellipsoid/mesh). `hertz_mindlin_field` unpacks the
first 9 **positionally**, so appending is safe but **never insert a column in the middle** — the
analytic baseline would silently read the wrong fields without raising.

## Ground truth: PhysX → IPC (important)

The **tangential/shear channel of PhysX deformable-body FEM does NOT converge** under mesh
refinement (model error, not discretization). We therefore use **IPC (libuipc via TacEx/GIPC)**
as the GT backend because its barrier contact lets us **measure and systematically reduce** the
contact error — which PhysX does not. This switch is a **framework decision (picking a GT whose
error is quantifiable), not a scientific claim** — TacEx already built GIPC to fix PhysX. Do not
frame "PhysX non-convergence" or "IPC converges" as paper findings. In particular **do not claim
"convergent friction" or "mesh-consistent tangential response"** — see the convergence map below.

Key facts about the IPC GT pipeline (`tacex_uipc_extract_shear.py`):
- Gel mesh = **structured tet box** (deterministic); indentor = **fan-from-center tet sphere**
  (deterministic). **Never use wildmeshing** (non-deterministic + irregular top surface breaks
  field sampling).
- **`d_hat` and `indentor_subdiv` are ONE knob, not two.** The indentor is a polyhedron; its
  geometric error is the facet sagitta ≈ edge²/(8R) (subdiv 2 → 5.3e-5 m, 3 → 1.3e-5, 4 → 3.4e-6,
  5 → 8.4e-7 at R=4.7mm). The barrier must resolve the sphere: **keep `d_hat / sagitta ≳ 5`**.
  Refining one while pinning the other produces *fake* non-monotone convergence curves — this
  wrecked every convergence study before 2026-07-13.
- **The solver is deterministic at a tight tolerance.** Run-to-run spread is a function of
  `velocity_tol`, not of the GPU: `3e-4` → **22.6%** on a full-slip frame, `1e-5` → **0.03–0.32%**.
  At `velocity_tol 1e-5`, **K=1 is correct** — replicate averaging is only papering over a loose
  tolerance.
- **Time is a refinement axis too.** Raising `shear_steps` at fixed `dt` is *not* refinement — it
  lowers the drive speed, i.e. changes the physics. True refinement halves `dt` **and** doubles all
  four step counts.
- **Production config (regen v2, 2026-07-13):** `gel_res 32, gel_nz 24, indentor_subdiv 4,
  d_hat 2.5e-5, eps_velocity 2.5e-5, velocity_tol 1e-5, contact_resistance 1e9, dt 0.005,
  press/settle/shear/shear-settle 80/20/160/20, K=1`. ~109 s/frame. **Always pass `--gel-nz`
  explicitly** — otherwise it defaults to a staircase (`round(lz/dx)` = 4 layers at res24).
- Driver modes: `--smoke`, `--single`, `--convergence`, `--batch`. `--batch` loops all
  frames × K reps in **one Isaac boot** (~23s boot amortised) → ~2.9× throughput.
- Production sweep: `infra/run_regen_v2_converged.sh` (wraps `infra/gen_uipc_sweep.sh`, which is
  env-var driven — **its bare defaults are `soft`/100/100 and will silently give you a different
  physical system**). → per-combo `data/uipc/sweep_v2_converged/combo_*/frame_*/rep_*/` then
  per-frame `uipc_gt_shear_avg.npz`. Single sequential stream beats sharding (one IPC sim already
  saturates the GPU). Concatenates to `data/uipc/shear_res32_nz24_sd4_converged.npz`, then
  `python -m novbts.research.fno.fem_benchmark --data <npz>`.
- The **pre-v2 dataset** (`shear_res24_avg_swept_REALISTIC_BC.npz`, what the paper currently cites)
  was generated at `gel_nz=4` and is **24–30% off in field shape** (40% on stick frames) against a
  refined reference. Do not quote its numbers as physical truth.

## Docker (Isaac Sim / IPC)

GT generation runs inside the Isaac Sim image `isaac-lab-tacex:latest` (libuipc built in via
`docker commit`, not yet Dockerfile-ized). Canonical run:
```
docker run --rm --gpus all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 \
  -v $PWD:/work --entrypoint /isaac-sim/python.sh isaac-lab-tacex:latest <driver> ...
```
Gotchas:
- Isaac swallows stdout → progress is written to a `fem_progress_uipc_*.txt` file; success markers
  like `SINGLE_UIPC_OK` / `SINGLE_UIPC_OK`.
- **Container exit codes are unreliable** → gate success on the npz file existing AND loading.
- **Container writes files as root.** To work with them on the host without sudo, chown via a
  root container: `docker run --rm -v $PWD:/work --entrypoint bash isaac-lab-tacex:latest -c \
  "chown -R $(id -u):$(id -g) /work/<dir>"`.

## Data & git

- `.gitignore` excludes `.venv*/ data/ runs/ logs/ *.pdf *.log __pycache__/ .claude/ codex/
  scratchpad/` plus transient Isaac progress markers (`fem_progress*.txt`, `sim_grasp_*progress.txt`)
  and the local `litellm-nvidia.yaml` — only code + `.md` docs are tracked. **Deleting data is NOT
  recoverable via git** (it's gitignored); FEM/IPC data is expensive to regenerate — be careful with
  deletes.
- Canonical PhysX shear GT: `data/fem/shear_fine.npz` + `shear_coarse.npz`; paired chunks in
  `data/fem/chunks/`. IPC GT under `data/uipc/`.

## Working notes

- Persistent project state across sessions lives in the auto-memory directory
  (`~/.claude/projects/.../memory/`, indexed by `MEMORY.md`) — read it for current phase status
  (phases 3–7), open problems, and the IPC GT build details. Codex handoff: `codex/NEXT_STEPS_phase7.md`.
- Do **not** auto-regenerate reports/PDFs after every discussion; only sync when explicitly asked.
- Reports and the KSE2026 paper draft are under `docs/` (incl. `docs/kse2026/`).
