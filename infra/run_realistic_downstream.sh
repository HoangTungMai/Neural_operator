#!/usr/bin/env bash
# Rerun every downstream experiment on the realistic UIPC dataset.
# Run only after gen_uipc_sweep.sh has produced the final 2520-frame NPZ.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-gate2/bin/python}"
DATA="${DATA:-data/uipc/shear_res24_avg_swept_REALISTIC.npz}"

"$PY" - "$DATA" <<'PY'
import os
import sys

import numpy as np

path = sys.argv[1]
if not os.path.isfile(path):
    raise SystemExit(f"missing production dataset: {path}")
d = np.load(path, allow_pickle=True)
n = int(np.asarray(d["disp"]).shape[0])
modes = np.asarray(d["mode"], dtype=np.int64).reshape(-1)
reps = np.asarray(d["n_replicates"], dtype=np.int64).reshape(-1)
gel_res = np.asarray(d["gel_res"], dtype=np.int64).reshape(-1)
eps = np.asarray(d["eps_velocity"], dtype=np.float64).reshape(-1)
velocity_tol = np.asarray(d["velocity_tol"], dtype=np.float64).reshape(-1)
d_hat = np.asarray(d["d_hat"], dtype=np.float64).reshape(-1)
contact_resistance = np.asarray(d["contact_resistance"], dtype=np.float64).reshape(-1)
sampling = np.asarray(d["marker_sampling"]).astype(str).reshape(-1)
tang_noise = np.asarray(d["rep_noise_tangential"], dtype=np.float64).reshape(-1)
split_test_size = int(np.asarray(d["split_test_size"]).reshape(-1)[0])
split_seed = int(np.asarray(d["split_shuffle_seed"]).reshape(-1)[0])
if n != 2520:
    raise SystemExit(f"expected 2520 frames, found {n}")
if set(np.unique(modes).tolist()) != {0, 1, 2, 3}:
    raise SystemExit(f"expected modes 0..3, found {np.unique(modes).tolist()}")
if not np.all(reps == 3):
    raise SystemExit(f"expected K=3 throughout, found {np.unique(reps).tolist()}")
if not np.all(gel_res == 24):
    raise SystemExit(f"expected gel_res=24 throughout, found {np.unique(gel_res).tolist()}")
if not np.allclose(eps, 2.5e-5):
    raise SystemExit(f"unexpected eps_velocity: {np.unique(eps).tolist()}")
if not np.allclose(velocity_tol, 1e-3):
    raise SystemExit(f"unexpected velocity_tol: {np.unique(velocity_tol).tolist()}")
if not np.allclose(d_hat, 1e-4):
    raise SystemExit(f"unexpected d_hat: {np.unique(d_hat).tolist()}")
if not np.allclose(contact_resistance, 1e9):
    raise SystemExit(f"unexpected contact_resistance: {np.unique(contact_resistance).tolist()}")
if set(sampling.tolist()) != {"bilinear"}:
    raise SystemExit(f"unexpected marker_sampling: {np.unique(sampling).tolist()}")
if split_test_size != 400 or split_seed != 2026:
    raise SystemExit(f"unexpected split metadata: test_size={split_test_size}, seed={split_seed}")
if set(np.unique(modes[-400:]).tolist()) != {0, 1, 2, 3}:
    raise SystemExit("the final 400-frame test split does not contain all four modes")
if not np.isfinite(tang_noise).all() or float(tang_noise[modes != 0].mean()) > 0.03:
    raise SystemExit(
        f"mean non-normal tangential replicate noise is "
        f"{float(tang_noise[modes != 0].mean()):.2%}, expected <=3%"
    )
print("REALISTIC_DATA_GATE_OK", {
    "frames": n,
    "mode_counts": np.bincount(modes, minlength=4).tolist(),
    "mean_k3_solve_time_s": float(np.asarray(d["solve_time_s"]).mean()),
})
PY

backup_once() {
  local src="$1"
  local stem="${src%.*}"
  local ext="${src##*.}"
  local dst="${stem}.PRE_REALISTIC.${ext}"
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

echo "REALISTIC_DOWNSTREAM_DONE"
