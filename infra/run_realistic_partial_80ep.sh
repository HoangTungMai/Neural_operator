#!/usr/bin/env bash
# Full-settings CPU rehearsal on a frozen partial realistic-GT snapshot.
# This validates/trains the complete downstream stack while GPU production runs.
# Results are rehearsal-only; final scientific results must be retrained on N=2520.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-gate2/bin/python}"
DATA="${DATA:-data/uipc/rehearsal/shear_realistic_partial.npz}"
N_TEST="${N_TEST:-200}"

"$PY" - "$DATA" "$N_TEST" <<'PY'
import sys
import numpy as np

path, n_test = sys.argv[1], int(sys.argv[2])
d = np.load(path, allow_pickle=True)
n = len(d["mode"])
if n <= n_test:
    raise SystemExit(f"snapshot N={n} must exceed n_test={n_test}")
if set(np.unique(d["mode"]).tolist()) != {0, 1, 2, 3}:
    raise SystemExit("snapshot must contain all four modes")
if not np.all(np.asarray(d["n_replicates"]) == 3):
    raise SystemExit("snapshot must be K=3 averaged")
print("PARTIAL_80EP_GATE_OK", {"N": n, "train": n - n_test, "test": n_test})
PY

"$PY" -m novbts.operator.fem_benchmark \
  --data "$DATA" --n-test "$N_TEST" --epochs 80 --clf-epochs 40 \
  --modes 12 --lr 0.001 --lambda-cls 0.1

"$PY" -m novbts.operator.vbts_baselines \
  --data "$DATA" --n-test "$N_TEST" --epochs 80 --modes 12 --lr 0.001 --ksize 31

"$PY" -m novbts.operator.diff_policy \
  --data "$DATA" --train-policy --task servo \
  --n-test "$N_TEST" --epochs 80 --modes 12 --lr 0.001 \
  --steps 300 --bs 128 --policy-lr 0.01 --lambda-reg 0 \
  --es-pop 32 --es-sigma 0.02 --log-every 10 --n-seeds 3

"$PY" -m novbts.sensor.build_sensor_dataset \
  --data "$DATA" \
  --px 160 --sensor-marker-side 11 --marker-placement pixel_even \
  --marker-pixel-fill 0.75 --marker-inset 0.06 \
  --working-dist 0.05 --fill 0.85 --sigma 1.35 \
  --dot-polarity dark --background 0.72 --contrast 0.58 \
  --saturate-dots --rt-n 120 --track-win 5 --sample-n-per-mode 3

"$PY" -m novbts.sensor.sensor_inverse_demo \
  --data "$DATA" \
  --n-test "$N_TEST" --epochs 80 --modes 12 --lr 0.001 \
  --px 160 --sensor-marker-side 11 --marker-placement pixel_even \
  --marker-pixel-fill 0.75 --marker-inset 0.06 \
  --working-dist 0.05 --sigma 1.35 --dot-polarity dark \
  --background 0.72 --contrast 0.58 --saturate-dots \
  --compare-n-per-mode 2 --steps 400 --opt-lr 0.05 \
  --inverse-n-per-mode 5 --inverse-restarts 8 \
  --inverse-min-shear-frac 0.01

"$PY" -m novbts.sensor.tactile_env \
  --demo --data "$DATA" \
  --n-test "$N_TEST" --epochs 80 --modes 12 --lr 0.001 \
  --sensor-side 11 --px 64 --reward-mode image --noise-read 0.02 \
  --steps 300 --bs 32 --policy-lr 0.01 \
  --gradcheck-batch 4 --preview-k 4

"$PY" -m novbts.report.make_kse_figs
mkdir -p "$NOVBTS_DOCS_DIR/kse2026/figs"
cp -a "$NOVBTS_RUNS_DIR/phase5/gt_vs_fno_samples.png" \
  "$NOVBTS_DOCS_DIR/kse2026/figs/sensor_gt_vs_fno.png"

echo "PARTIAL_80EP_REHEARSAL_DONE"
