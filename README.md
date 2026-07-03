# novbts — Neural-operator surrogate for a vision-based tactile sensor

An **FNO** learns marker displacement fields as a fast surrogate that replaces an
expensive contact solver, for downstream RL/control. The current paper headline
numbers are produced on the corrected-BC realistic **IPC/UIPC** thin-gel dataset:

`data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz`

The headline benchmark is **not** `operator/field2field.py`. It is
`operator/fem_benchmark.py` run with `--field-model lr_fno`, which loads the
IPC/UIPC `.npz` directly and reports the paper metrics (for example LR-FNO
`0.060 / 1.5 deg` and `9.0x` over the per-point MLP). The main reported model is
**LR-FNO** (`operator/hybrid_fno.py`: FNO trunk + lightweight local refinement,
U-FNO family, Wen et al. 2022); the plain FNO trunk is kept as an ablation.

`operator/field2field.py` is retained as an analytic Hertz--Mindlin
proof-of-concept of the same field-to-field framing. It should not be used to
reproduce the paper headline table.

## Canonical Corrected-BC Results

Use this table as the source of truth for the current KSE submission. Older
reports in `docs/` may describe superseded soft-BC or pre-reground runs.

| Item | Canonical artifact |
|---|---|
| Ground truth | `data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz` |
| Acceptance checker | `infra/verify_bc_reground.py` |
| Full downstream rerun | `infra/run_bc_downstream.sh` (plain FNO) / `infra/run_lrfno_downstream.sh` (LR-FNO, current) |
| RQ1/RQ2/RQ3 | `runs/phase3_fem/benchmark.json` |
| Baseline bakeoff | `runs/phase3_fem/vbts_baselines.json` |
| 5-seed architecture benchmark | `runs/phase3_fem/hybrid_multiseed.json` |
| Control | `runs/phase4/policy_servo.json` |
| Sensor build / inverse | `runs/phase5/sensor_build.json`, `runs/phase5/sensor_inverse_multiframe.json` |
| Environment demo | `runs/phase6/env_demo.json` |
| Paper | `docs/kse2026/main.tex`, `docs/kse2026/main.pdf` |

Current headline numbers are LR-FNO `rel-L2=0.060` (5-seed `0.0590±0.0014`,
paired t=4.72 vs the plain FNO trunk), per-point MLP `rel-L2=0.545` (`9.0x`
ratio), single-solve IPC/UIPC speedup `29,182x`, and adaptive raw `K=5`
corrected-BC targets. Prior-lineage numbers (plain-FNO main-model era
`rel-L2=0.074`/`7.35x`/`41,253x`; soft-BC `rel-L2=0.041`/`12.14x`/`82,827x`/`K=3`)
are retained only as provenance — plain-FNO downstream artifacts are backed up
as `*.FNO_MAIN.*`.

## Layout

```
src/novbts/
  models.py              FNO / MLP / DeepONet / SpectralConv2d (shared defs)
  paths.py               central path config (ROOT/DATA/RUNS/DOCS)
  groundtruth/
    hertz_mindlin.py     analytic Hertz + Cattaneo–Mindlin GT + validator
    data_gen.py          analytic dataset generator (train/test/OOD splits)
    isaac_extract_normal.py   PhysX-FEM GT, normal indentation  (runs in Docker)
    isaac_extract_shear.py    PhysX-FEM GT, shear/slip          (runs in Docker)
    tacex_uipc_extract_shear.py IPC/UIPC thin-gel shear generator (TacEx-style)
  operator/
    fem_benchmark.py     PAPER HEADLINE benchmark on realistic IPC/UIPC .npz (--field-model lr_fno)
    hybrid_fno.py        LR-FNO (main model) / U-FNO-style defs + 5-seed multi-model benchmark
    vbts_baselines.py    VBTS/classical/neural baseline bakeoff on IPC/UIPC GT
    field2field.py       analytic field→field PoC (Hertz--Mindlin, not paper headline)
    param2field.py       param→field framing (ablation) + slip heads
    eval_rq.py           RQ1–RQ3 evaluation (accuracy / generalization / speed)
    fem_train_compare.py train on coarse- vs fine-mesh FEM GT, eval on fine
  validation/
    validate_gt.py       PhysX-FEM vs Hertz–Mindlin agreement
    validate_shear.py    shear GT sanity / saturation signal
    compare_shear.py     coarse vs fine shear GT (stick-radius / resolution)
  report/make_pdf.py     render the Phase-3 report to PDF (-> docs/)

infra/                   Isaac/IPC generation scripts + realistic acceptance checker
scripts/archive/         frozen one-off probes & superseded scripts
docs/                    reports (.md live, .pdf generated); docs/archive/ older gates
data/    (gitignored)    uipc/ realistic GT  ·  analytic/ PoC  ·  fem/ legacy PhysX
runs/    (gitignored)    training/eval outputs  ·  convergence/ (mesh study)
logs/    (gitignored)
```

## Setup

```bash
.venv-gate2/bin/pip install -e .          # editable install of the novbts package
```

Paths resolve from the repo root via `novbts.paths`, so modules run from any CWD.

## Run (no Isaac needed)

```bash
python -m novbts.operator.fem_benchmark --field-model lr_fno \
  --data data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz
                                                # paper headline RQ1/RQ2/RQ3 on IPC/UIPC GT
python -m novbts.operator.hybrid_fno            # 5-seed FNO/U-Net/LR-FNO/U-FNO benchmark
python -m novbts.operator.vbts_baselines \
  --data data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz
                                                # paper baseline bakeoff on same IPC/UIPC split
python -m novbts.operator.field2field           # analytic field→field PoC only
python -m novbts.report.make_kse_figs           # regenerate KSE figures from current runs/
python infra/verify_bc_reground.py              # acceptance check for data/downstream/paper
```

## Data

- `data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz` — current IPC/UIPC paper GT:
  thin gel `20 x 20 x 3 mm`, res-24, marker grid `32 x 32`, `N=2520`, corrected
  fixed-bottom BC, adaptive `K<=5` robust-averaged replicates, deterministic
  `2120/400` train/test split. (`..._REALISTIC.npz` is the soft-BC predecessor,
  kept as provenance; its tangential channel carries a rigid-drift artifact.)
- `data/analytic/` — Hertz–Mindlin train (16k) / val / test / OOD splits, side-32 marker grid.
  Used by `field2field.py` as a proof-of-concept, not as the paper headline GT.
- `data/fem/normal.npz` — PhysX-FEM ground truth, normal indentation (40 frames).
- `data/fem/shear_fine.npz`, `data/fem/shear_coarse.npz` — shear GT at 50×50×20 mm,
  res-24 fine vs default coarse mesh (200 frames each, side-32 probe grid). These are
  legacy PhysX artifacts and are not the current paper headline data.
- `data/fem/chunks/` — raw per-seed FEM runs (`fem_{fine,coarse}_s43..s46`), kept to
  rebuild a strictly *paired* fine-vs-coarse set if needed.

IPC/UIPC ground truth is generated by the TacEx-style pipeline and aggregated before
downstream training/evaluation. PhysX/Isaac scripts remain in the tree for legacy
experiments and archived reports.
