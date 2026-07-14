#!/usr/bin/env bash
# Small Phase 7 boundary-condition sweep for the UIPC bottom constraint.
set -euo pipefail
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
OUT_ROOT="${OUT_ROOT:-data/uipc/phase7_bc_sweep}"
SCRIPT="/work/src/novbts/research/groundtruth/tacex_uipc_extract_shear.py"
STRENGTHS="${STRENGTHS:-100 300 1000 3000 10000}"
INCLUDE_FIXED="${INCLUDE_FIXED:-1}"

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
INDENTOR_CONSTRAINT_STRENGTH="${INDENTOR_CONSTRAINT_STRENGTH:-100}"
FIXED_INDENTOR_CONSTRAINT_STRENGTH="${FIXED_INDENTOR_CONSTRAINT_STRENGTH:-$INDENTOR_CONSTRAINT_STRENGTH}"
PRESS_STEPS="${PRESS_STEPS:-40}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
SHEAR_STEPS="${SHEAR_STEPS:-80}"
SHEAR_SETTLE="${SHEAR_SETTLE:-10}"

mkdir -p "$OUT_ROOT"

"$PY" - "$OUT_ROOT/cases.json" "$OUT_ROOT/rows.txt" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
rows_path = Path(sys.argv[2])
mu = float(os.environ.get("MU", "0.6"))
shear_scale = float(os.environ.get("SHEAR_SCALE", "0.001"))

spec = [
    ("normal_mid", 0.00055, 0.0, 0.0),
    ("partial_mid", 0.00055, 0.80, math.radians(60.0)),
    ("full_mid", 0.00055, 1.30, math.radians(60.0)),
    ("full_max", 0.00075, 1.30, math.radians(60.0)),
]
rows = []
with rows_path.open("w") as f:
    for frame, (name, depth, g, theta) in enumerate(spec):
        shear_mag = g * mu * shear_scale
        sx = shear_mag * math.cos(theta)
        sy = shear_mag * math.sin(theta)
        f.write(f"{frame} {depth:.9g} {g:.9g} {sx:.9g} {sy:.9g} linear\n")
        rows.append({
            "frame": frame,
            "case": name,
            "depth_m": depth,
            "drive_ratio": g,
            "sx_m": sx,
            "sy_m": sy,
            "load_mode": "linear",
        })
json.dump(rows, open(cases_path, "w"), indent=2)
print(f"wrote {cases_path} and {rows_path}")
PY

for strength in $STRENGTHS; do
  label="${strength//./p}"
  label="${label//+}"
  out_dir="$OUT_ROOT/strength_${label}/combo_000"
  progress="$OUT_ROOT/strength_${label}.log"
  mkdir -p "$out_dir"
  echo "PHASE7_BC_SWEEP strength=$strength"
  rtk proxy docker run --rm --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 \
    -v "$PWD":/work --entrypoint /isaac-sim/python.sh "$IMG" \
    "$SCRIPT" \
    --batch \
    --batch-rows "/work/$OUT_ROOT/rows.txt" \
    --batch-reps 1 \
    --out "/work/$out_dir" \
    --gel-xy "$GEL_XY" --gel-z "$GEL_Z" --gel-res "$GEL_RES" \
    --indentor-r "$INDENTOR_R" --mu "$MU" --youngs "$YOUNGS" \
    --depth 0.00055 --shear 0.0004 \
    --eps-velocity "$EPS_VELOCITY" --velocity-tol "$VELOCITY_TOL" \
    --d-hat "$D_HAT" --contact-resistance "$CONTACT_RESISTANCE" \
    --press-steps "$PRESS_STEPS" --settle-steps "$SETTLE_STEPS" \
    --shear-steps "$SHEAR_STEPS" --shear-settle "$SHEAR_SETTLE" \
    --marker-side 32 \
    --gel-constraint-strength "$strength" \
    --indentor-constraint-strength "$INDENTOR_CONSTRAINT_STRENGTH" \
    --progress-file "/work/$progress"
done

if [[ "$INCLUDE_FIXED" == "1" ]]; then
  out_dir="$OUT_ROOT/fixed/combo_000"
  progress="$OUT_ROOT/fixed.log"
  mkdir -p "$out_dir"
  echo "PHASE7_BC_SWEEP gel-bottom-bc=fixed indentor_strength=$FIXED_INDENTOR_CONSTRAINT_STRENGTH"
  rtk proxy docker run --rm --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 \
    -v "$PWD":/work --entrypoint /isaac-sim/python.sh "$IMG" \
    "$SCRIPT" \
    --batch \
    --batch-rows "/work/$OUT_ROOT/rows.txt" \
    --batch-reps 1 \
    --out "/work/$out_dir" \
    --gel-xy "$GEL_XY" --gel-z "$GEL_Z" --gel-res "$GEL_RES" \
    --indentor-r "$INDENTOR_R" --mu "$MU" --youngs "$YOUNGS" \
    --depth 0.00055 --shear 0.0004 \
    --eps-velocity "$EPS_VELOCITY" --velocity-tol "$VELOCITY_TOL" \
    --d-hat "$D_HAT" --contact-resistance "$CONTACT_RESISTANCE" \
    --press-steps "$PRESS_STEPS" --settle-steps "$SETTLE_STEPS" \
    --shear-steps "$SHEAR_STEPS" --shear-settle "$SHEAR_SETTLE" \
    --marker-side 32 \
    --gel-bottom-bc fixed \
    --indentor-constraint-strength "$FIXED_INDENTOR_CONSTRAINT_STRENGTH" \
    --progress-file "/work/$progress"
fi

rtk proxy docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
  -c "chown -R $(id -u):$(id -g) /work/$OUT_ROOT" >/dev/null 2>&1

rtk proxy "$PY" -m novbts.research.groundtruth.uniform_shift_diagnostic \
  --root "$OUT_ROOT" \
  --cases-json "$OUT_ROOT/cases.json" \
  --out-json "$OUT_ROOT/summary.json" \
  --out-md "$OUT_ROOT/summary.md"

echo "PHASE7_BC_SWEEP_DONE $OUT_ROOT/summary.md"
