#!/usr/bin/env bash
# Rerun every downstream experiment on the corrected-BC UIPC dataset (Phase D of
# the corrected-BC production reground). Mirrors run_realistic_downstream.sh but:
#   * gates the dataset through infra/verify_bc_reground.py (adaptive-K aware)
#     instead of the K=3 soft-BC assertions,
#   * backs up prior artifacts once as *.PRE_BC.* before overwriting.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-gate2/bin/python}"
DATA="${DATA:-data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz}"

"$PY" infra/verify_bc_reground.py

backup_once() {
  local src="$1"
  local stem="${src%.*}"
  local ext="${src##*.}"
  local dst="${stem}.PRE_BC.${ext}"
  if [[ -e "$src" && ! -e "$dst" ]]; then
    cp -a "$src" "$dst"
  fi
}

for artifact in \
  runs/phase3_fem/benchmark.json \
  runs/phase3_fem/vbts_baselines.json \
  runs/phase4/policy_servo.json \
  runs/phase4/policy_servo_curve.png \
  runs/phase5/sensor_build.json \
  runs/phase5/sensor_compat.json \
  runs/phase5/sensor_inverse.json \
  runs/phase5/sensor_inverse_multiframe.json \
  runs/phase5/sensor_compare.json \
  runs/phase5/preview.png \
  runs/phase5/test_samples.png \
  runs/phase5/gt_vs_fno_samples.png \
  runs/phase6/env_demo.json \
  runs/phase6/env_demo.png \
  docs/kse2026/figs/fidelity_speed.png \
  docs/kse2026/figs/policy_servo_curve.png \
  docs/kse2026/figs/sensor_gt_vs_fno.png
do
  backup_once "$artifact"
done

"$PY" -m novbts.operator.fem_benchmark \
  --data "$DATA" --n-test 400 --epochs 80 --clf-epochs 40 \
  --modes 12 --lr 0.001 --lambda-cls 0.1

"$PY" -m novbts.operator.vbts_baselines \
  --data "$DATA" --n-test 400 --epochs 80 --modes 12 --lr 0.001 --ksize 31

"$PY" -m novbts.operator.diff_policy \
  --data "$DATA" --train-policy --task servo \
  --n-test 400 --epochs 80 --modes 12 --lr 0.001 \
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
  --n-test 400 --epochs 80 --modes 12 --lr 0.001 \
  --px 160 --sensor-marker-side 11 --marker-placement pixel_even \
  --marker-pixel-fill 0.75 --marker-inset 0.06 \
  --working-dist 0.05 --sigma 1.35 --dot-polarity dark \
  --background 0.72 --contrast 0.58 --saturate-dots \
  --compare-n-per-mode 1 --steps 400 --opt-lr 0.05 \
  --inverse-n-per-mode 5 --inverse-restarts 8 \
  --inverse-min-shear-frac 0.01

"$PY" -m novbts.sensor.tactile_env \
  --demo --data "$DATA" \
  --n-test 400 --epochs 80 --modes 12 --lr 0.001 \
  --sensor-side 11 --px 64 --reward-mode image --noise-read 0.02 \
  --steps 300 --bs 32 --policy-lr 0.01 \
  --gradcheck-batch 4 --preview-k 4

"$PY" -m novbts.report.make_kse_figs
cp -a runs/phase5/gt_vs_fno_samples.png docs/kse2026/figs/sensor_gt_vs_fno.png

echo "BC_DOWNSTREAM_DONE"
