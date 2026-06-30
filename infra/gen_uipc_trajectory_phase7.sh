#!/usr/bin/env bash
# Phase 7 trajectory/loading-history GT on realistic UIPC.
#
# Produces a small endpoint-controlled dataset where each endpoint is repeated
# with three loading paths (linear/ortho/reverse). Output keeps the legacy
# trajectory schema expected by novbts.operator.loading_history:
#   params, coords, disp, mode, disp_traj, traj_fracs, load_mode, load_mode_names
set -euo pipefail
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
SCRIPT="/work/src/novbts/groundtruth/tacex_uipc_extract_shear.py"
OUT_ROOT="${OUT_ROOT:-data/uipc/trajectory_phase7}"
N_ENDPOINTS="${1:-12}"
KREPS="${2:-3}"
TEST_SIZE="${3:-9}"

ROWS_DIR="$OUT_ROOT/_rows"
ROWS="$ROWS_DIR/combo_000.rows"
SWEEP_DIR="$OUT_ROOT/sweep"
FINAL_NPZ="$OUT_ROOT/shear_res24_traj_REALISTIC.npz"
PROGRESS="$OUT_ROOT/progress_combo_000.log"

mkdir -p "$ROWS_DIR" "$SWEEP_DIR"

"$PY" - "$ROWS" "$N_ENDPOINTS" <<'PY'
import math
import sys
from pathlib import Path

rows = Path(sys.argv[1])
n = int(sys.argv[2])
load_modes = ("linear", "ortho", "reverse")

# Endpoint set spans stick/partial/full while keeping identical endpoints across
# load modes. Units and param conventions mirror the realistic static sweep.
depths = [0.00020, 0.00035, 0.00050, 0.00065]
drive = [0.20, 0.55, 0.90]
rows.parent.mkdir(parents=True, exist_ok=True)
with rows.open("w") as f:
    frame = 0
    for i in range(n):
        depth = depths[i % len(depths)]
        g = drive[i % len(drive)]
        theta = 2.0 * math.pi * (i / max(n, 1))
        mu = 0.6
        shear_mag = g * mu * 0.001
        sx = shear_mag * math.cos(theta)
        sy = shear_mag * math.sin(theta)
        for lm in load_modes:
            f.write(f"{frame} {depth:.9g} {g:.9g} {sx:.9g} {sy:.9g} {lm}\n")
            frame += 1
print(f"wrote {rows} with {n * len(load_modes)} frames")
PY

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 \
  -v "$PWD":/work --entrypoint /isaac-sim/python.sh "$IMG" \
  "$SCRIPT" \
  --batch \
  --batch-rows "/work/$ROWS" \
  --batch-reps "$KREPS" \
  --out "/work/$SWEEP_DIR/combo_000" \
  --gel-xy 0.020 --gel-z 0.003 --gel-res 24 \
  --indentor-r 0.004 --mu 0.6 --youngs 1.0e5 \
  --depth 0.00035 --shear 0.0004 \
  --eps-velocity 2.5e-5 --velocity-tol 1e-3 \
  --d-hat 1e-4 --contact-resistance 1e9 \
  --press-steps 40 --settle-steps 10 --shear-steps 80 --shear-settle 10 \
  --marker-side 32 \
  --save-trajectory --traj-steps 8 \
  --progress-file "/work/$PROGRESS"

docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
  -c "chown -R $(id -u):$(id -g) /work/$OUT_ROOT" >/dev/null 2>&1

"$PY" -m novbts.groundtruth.aggregate_uipc_replicates \
  --sweep-dir "$SWEEP_DIR" \
  --out "$FINAL_NPZ" \
  --expect-reps "$KREPS" \
  --test-size "$TEST_SIZE" \
  --mode-shear-scale 0.001

"$PY" -m novbts.operator.loading_history \
  --data "$FINAL_NPZ" \
  --n-test "$TEST_SIZE" \
  --epochs 120 \
  --modes 12 \
  --lr 0.001

"$PY" -m novbts.sensor.temporal \
  --data "$FINAL_NPZ" \
  --out-dir phase7

"$PY" -m novbts.sensor.temporal_compare \
  --data "$FINAL_NPZ" \
  --fno-data data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  --fno-epochs 80 \
  --modes 12 \
  --out-dir phase7

echo "PHASE7_TRAJECTORY_DONE $FINAL_NPZ"
