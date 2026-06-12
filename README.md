# novbts — Neural-operator surrogate for a vision-based tactile sensor

An **FNO** learns marker displacement fields as a fast surrogate that replaces an
expensive contact solver, for downstream RL. Ground truth comes from **PhysX
Deformable-Body FEM** (Isaac Sim); **Hertz–Mindlin** provides an analytic
validator. The headline result is the *field→field* operator framing, where the
FNO decisively beats a per-point MLP (non-local elastic response).

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
  operator/
    field2field.py       HEADLINE field→field operator (FNO vs MLP)
    param2field.py       param→field framing (ablation) + slip heads
    eval_rq.py           RQ1–RQ3 evaluation (accuracy / generalization / speed)
    fem_train_compare.py train on coarse- vs fine-mesh FEM GT, eval on fine
  validation/
    validate_gt.py       PhysX-FEM vs Hertz–Mindlin agreement
    validate_shear.py    shear GT sanity / saturation signal
    compare_shear.py     coarse vs fine shear GT (stick-radius / resolution)
  report/make_pdf.py     render the Phase-3 report to PDF (-> docs/)

infra/                   Dockerfile.fem, setup_isaac.sh (Isaac Sim env)
scripts/archive/         frozen one-off probes & superseded scripts
docs/                    reports (.md live, .pdf generated); docs/archive/ older gates
data/    (gitignored)    analytic/ (Hertz–Mindlin)  ·  fem/ (PhysX GT + chunks/)
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
python -m novbts.operator.field2field        # headline field→field FNO vs MLP
python -m novbts.operator.fem_train_compare   # coarse- vs fine-mesh FEM GT
python -m novbts.operator.eval_rq             # RQ1–RQ3 tables + fidelity-speed
python -m novbts.report.make_pdf              # regenerate docs/bao_cao_giai_doan3.pdf
```

## Data

- `data/analytic/` — Hertz–Mindlin train (16k) / val / test / OOD splits, side-32 marker grid.
- `data/fem/normal.npz` — PhysX-FEM ground truth, normal indentation (40 frames).
- `data/fem/shear_fine.npz`, `data/fem/shear_coarse.npz` — shear GT at 50×50×20 mm,
  res-24 fine vs default coarse mesh (200 frames each, side-32 probe grid).
- `data/fem/chunks/` — raw per-seed FEM runs (`fem_{fine,coarse}_s43..s46`), kept to
  rebuild a strictly *paired* fine-vs-coarse set if needed.

FEM ground truth is generated inside the `isaac-lab-fem` Docker image; the extractor
scripts run standalone in the container (no `novbts` import) and log progress to a
mounted file because Isaac swallows stdout.
