#!/usr/bin/env bash
# Phase 7 trajectory/loading-history GT on realistic UIPC.
#
# Produces a small endpoint-controlled dataset where each endpoint is repeated
# with three loading paths (linear/ortho/reverse). Output keeps the legacy
# trajectory schema expected by novbts.research.fno.loading_history:
#   params, coords, disp, mode, disp_traj, traj_fracs, load_mode, load_mode_names
set -euo pipefail
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
SCRIPT="/work/src/novbts/research/groundtruth/tacex_uipc_extract_shear.py"
OUT_ROOT="${OUT_ROOT:-data/uipc/trajectory_phase7}"
DATA_TAG="${DATA_TAG:-REALISTIC}"
RUN_OUT_DIR="${RUN_OUT_DIR:-phase7}"
GEL_XY="${GEL_XY:-0.020}"
GEL_Z="${GEL_Z:-0.003}"
GEL_RES="${GEL_RES:-24}"
INDENTOR_R="${INDENTOR_R:-0.004}"
MU="${MU:-0.6}"
YOUNGS="${YOUNGS:-1.0e5}"
SHEAR_SCALE="${SHEAR_SCALE:-0.001}"
EPS_VELOCITY="${EPS_VELOCITY:-2.5e-5}"
VELOCITY_TOL="${VELOCITY_TOL:-1e-3}"
D_HAT="${D_HAT:-1e-4}"
CONTACT_RESISTANCE="${CONTACT_RESISTANCE:-1e9}"
GEL_BOTTOM_BC="${GEL_BOTTOM_BC:-soft}"
GEL_CONSTRAINT_STRENGTH="${GEL_CONSTRAINT_STRENGTH:-100}"
INDENTOR_CONSTRAINT_STRENGTH="${INDENTOR_CONSTRAINT_STRENGTH:-100}"
PRESS_STEPS="${PRESS_STEPS:-40}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
SHEAR_STEPS="${SHEAR_STEPS:-80}"
SHEAR_SETTLE="${SHEAR_SETTLE:-10}"
TRAJ_STEPS="${TRAJ_STEPS:-8}"
N_ENDPOINTS="${1:-12}"
KREPS="${2:-3}"
TEST_SIZE="${3:-9}"

ROWS_DIR="$OUT_ROOT/_rows"
ROWS="$ROWS_DIR/combo_000.rows"
SWEEP_DIR="$OUT_ROOT/sweep"
FINAL_NPZ="$OUT_ROOT/shear_res24_traj_${DATA_TAG}.npz"
PROGRESS="$OUT_ROOT/progress_combo_000.log"

mkdir -p "$ROWS_DIR" "$SWEEP_DIR"

"$PY" - "$ROWS" "$N_ENDPOINTS" <<'PY'
import os
import math
import sys
from pathlib import Path

rows = Path(sys.argv[1])
n = int(sys.argv[2])
load_modes = ("linear", "ortho", "reverse")

# Endpoint set spans stick/partial/full while keeping identical endpoints across
# load modes. Units and param conventions mirror the realistic static sweep.
# Defaults are intentionally stronger than the first pilot so the temporal GIF
# contains full-slip frames with visible marker flow.
depths = [float(x) for x in os.environ.get(
    "DEPTH_LEVELS", "0.00035,0.00055,0.00075"
).split(",")]
drive = [float(x) for x in os.environ.get(
    "DRIVE_LEVELS", "0.30,0.80,1.30"
).split(",")]
rows.parent.mkdir(parents=True, exist_ok=True)
with rows.open("w") as f:
    frame = 0
    for i in range(n):
        depth = depths[i % len(depths)]
        g = drive[i % len(drive)]
        theta = 2.0 * math.pi * (i / max(n, 1))
        mu = float(os.environ.get("MU", "0.6"))
        shear_scale = float(os.environ.get("SHEAR_SCALE", "0.001"))
        shear_mag = g * mu * shear_scale
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
  --gel-xy "$GEL_XY" --gel-z "$GEL_Z" --gel-res "$GEL_RES" \
  --indentor-r "$INDENTOR_R" --mu "$MU" --youngs "$YOUNGS" \
  --depth 0.00035 --shear 0.0004 \
  --eps-velocity "$EPS_VELOCITY" --velocity-tol "$VELOCITY_TOL" \
  --d-hat "$D_HAT" --contact-resistance "$CONTACT_RESISTANCE" \
  --press-steps "$PRESS_STEPS" --settle-steps "$SETTLE_STEPS" --shear-steps "$SHEAR_STEPS" --shear-settle "$SHEAR_SETTLE" \
  --marker-side 32 \
  --gel-bottom-bc "$GEL_BOTTOM_BC" \
  --gel-constraint-strength "$GEL_CONSTRAINT_STRENGTH" \
  --indentor-constraint-strength "$INDENTOR_CONSTRAINT_STRENGTH" \
  --save-trajectory --traj-steps "$TRAJ_STEPS" \
  --progress-file "/work/$PROGRESS"

docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
  -c "chown -R $(id -u):$(id -g) /work/$OUT_ROOT" >/dev/null 2>&1

"$PY" -m novbts.research.groundtruth.aggregate_uipc_replicates \
  --sweep-dir "$SWEEP_DIR" \
  --out "$FINAL_NPZ" \
  --expect-reps "$KREPS" \
  --test-size "$TEST_SIZE" \
  --mode-shear-scale "$SHEAR_SCALE"

"$PY" -m novbts.research.fno.loading_history \
  --data "$FINAL_NPZ" \
  --n-test "$TEST_SIZE" \
  --epochs 120 \
  --modes 12 \
  --lr 0.001

"$PY" -m novbts.simulation.sensor.temporal \
  --data "$FINAL_NPZ" \
  --out-dir "$RUN_OUT_DIR"

"$PY" -m novbts.simulation.sensor.temporal_compare \
  --data "$FINAL_NPZ" \
  --fno-data data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  --fno-epochs 80 \
  --modes 12 \
  --out-dir "$RUN_OUT_DIR"

echo "PHASE7_TRAJECTORY_DONE $FINAL_NPZ"
