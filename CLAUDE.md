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
  modules run from any CWD. **No `sys.path` hacks** — always run as modules:
  - `.venv-gate2/bin/python -m novbts.operator.field2field`  (headline field→field FNO)
  - `python -m novbts.operator.fem_train_compare` / `.eval_rq` / `.fem_benchmark`
  - `python -m novbts.report.make_pdf`
- There is no formal test suite / linter; "tests" here are smoke runs of the drivers (e.g. the
  `--smoke` flag on GT drivers) and the benchmark/eval modules.

## Package layout (big picture)

`src/novbts/`
- `paths.py` — single source of truth for all data/run/docs locations.
- `groundtruth/` — generators for ground-truth deformation fields:
  - Analytic: `hertz_mindlin`, `data_gen`.
  - PhysX FEM (Isaac Sim): `isaac_extract_normal`, `isaac_extract_shear` — **self-contained,
    run standalone inside Docker, do NOT import novbts**.
  - IPC/GIPC (libuipc/TacEx): `tacex_uipc_extract_shear.py` (current GT backend, see below) +
    `aggregate_uipc_replicates.py`, `aggregate_uipc_convergence.py`.
- `operator/` — the FNO and training/eval: `field2field` (headline), `param2field`,
  `eval_rq`, `fem_train_compare`, `fem_benchmark`.
- `models/` — FNO / MLP / DeepONet / SpectralConv2d definitions.
- `sensor/` — differentiable marker-dot VBTS renderer (`tactile_env.py`, camera + dot render).
- `policy/` (`diff_policy.py`) — control policy learned through the differentiable FNO (Phase 4).
- `report/` — PDF/slide generators (`make_pdf`, `make_summary_pdf`, `make_slides`, `make_phase5_pdf`).
- `validation/` — GT validation/comparison utilities.
Dead code / PoCs live in `scripts/archive/` (NOT in the package). Infra (Dockerfiles, sweep
shell scripts) in `infra/`.

## Ground truth: PhysX → IPC (important)

The **tangential/shear channel of PhysX deformable-body FEM does NOT converge** under mesh
refinement (model error, not discretization). We therefore use **IPC (libuipc via TacEx/GIPC)**
as the GT backend because its barrier+lagged-friction contact converges. This switch is a
**framework decision (picking a trustworthy GT), not a scientific claim** — TacEx already built
GIPC to fix PhysX. Do not frame "PhysX non-convergence" or "IPC converges" as paper findings.

Key facts about the IPC GT pipeline (`tacex_uipc_extract_shear.py`):
- Gel mesh = **structured tet box** (deterministic); indentor = **fan-from-center tet sphere**
  (deterministic). **Never use wildmeshing** (non-deterministic + irregular top surface breaks
  field sampling).
- libuipc **GPU runs are NOT deterministic** run-to-run (tangential noise ~2.4% at res24, worse
  at higher res). Mitigation: **average K replicates** (noise /√K). K=3 is the default sweet spot.
- Use **res24 / eps_velocity 0.001** for production; do not refine past res24 for the tangential
  channel (noise grows).
- Driver modes: `--smoke`, `--single`, `--convergence`, `--batch`. `--batch` loops all
  frames × K reps in **one Isaac boot** (~23s boot amortised) → ~2.9× throughput.
- Production sweep: `infra/gen_uipc_sweep.sh <NCOMBOS> <FRAMES> <K_REPS> [START END]`
  → per-combo `data/uipc/sweep/combo_*/frame_*/rep_*/` then per-frame averaged
  `uipc_gt_shear_avg.npz`. Single sequential stream is faster than sharding (one IPC sim already
  saturates the GPU). Final dataset concatenates to `data/uipc/shear_res24_avg_swept.npz`, then
  `python -m novbts.operator.fem_benchmark --data <npz>`.

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

- `.gitignore` excludes `.venv*/ data/ runs/ logs/ *.pdf __pycache__/ .claude/` — only code + `.md`
  docs are tracked. **Deleting data is NOT recoverable via git** (it's gitignored); FEM/IPC data is
  expensive to regenerate — be careful with deletes.
- Canonical PhysX shear GT: `data/fem/shear_fine.npz` + `shear_coarse.npz`; paired chunks in
  `data/fem/chunks/`. IPC GT under `data/uipc/`.

## Working notes

- Persistent project state across sessions lives in the auto-memory directory
  (`~/.claude/projects/.../memory/`, indexed by `MEMORY.md`) — read it for current phase status
  (phases 3–7), open problems, and the IPC GT build details. Codex handoff: `codex/NEXT_STEPS_phase7.md`.
- Do **not** auto-regenerate reports/PDFs after every discussion; only sync when explicitly asked.
- Reports and the KSE2026 paper draft are under `docs/` (incl. `docs/kse2026/`).
