#!/usr/bin/env bash
# Geometry-OOD UIPC sweep. Tier 1 supports:
#   cylinder    : flat circular punch, R sampled in the production sphere range
#   sphere_oodR : sphere with R outside the production range (8/9/10 mm)
#
# Usage:
#   bash infra/gen_uipc_geom_ood.sh cylinder 6 25 3
#   bash infra/gen_uipc_geom_ood.sh sphere_oodR 6 25 3
set -u
cd "$(dirname "$0")/.."

GEOM_REQ="${1:-cylinder}"
NCOMBOS="${2:-6}"
FRAMES="${3:-25}"
KREPS="${4:-3}"
START_COMBO="${5:-0}"
END_COMBO="${6:-$((NCOMBOS - 1))}"

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
SCRIPT=/work/src/novbts/groundtruth/tacex_uipc_extract_shear.py
PROD_NPZ="${PROD_NPZ:-data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz}"
OOD_ROOT="${OOD_ROOT:-data/uipc/geom_ood}"
SWEEP_DIR="${SWEEP_DIR:-$OOD_ROOT/$GEOM_REQ/sweep}"
OUT_DATA="${OUT_DATA:-$OOD_ROOT/$GEOM_REQ/${GEOM_REQ}_avg.npz}"
ROWS_DIR="$SWEEP_DIR/_rows"
SHARD="${START_COMBO}_${END_COMBO}"
PROG="/work/fem_progress_uipc_geom_${GEOM_REQ}_${SHARD}.txt"
TEST_SIZE="${TEST_SIZE:-100}"

case "$GEOM_REQ" in
  cylinder) DRIVER_GEOM="cylinder" ;;
  sphere_oodR) DRIVER_GEOM="sphere" ;;
  *) echo "unsupported Tier-1 geometry: $GEOM_REQ (expected cylinder|sphere_oodR)" >&2; exit 2 ;;
esac

# Pull the production gel/BC/IPC knobs from the checked GT file unless the caller
# intentionally overrides them in the environment.
eval "$($PY - "$PROD_NPZ" <<'PYEOF'
import sys, numpy as np
path = sys.argv[1]
d = np.load(path, allow_pickle=True)
def first(key, default):
    if key not in d.files:
        return default
    a = np.asarray(d[key]).reshape(-1)
    return a[0].item()
params = np.asarray(d["params"])
print(f': "${{GEL_RES:={int(first("gel_res", 24))}}}"')
print(f': "${{EPS_VELOCITY:={float(first("eps_velocity", 0.000025))}}}"')
print(f': "${{D_HAT:={float(first("d_hat", 0.0001))}}}"')
print(f': "${{CONTACT_RESISTANCE:={float(first("contact_resistance", 1.0e9))}}}"')
print(f': "${{VELOCITY_TOL:={float(first("velocity_tol", 0.001))}}}"')
print(f': "${{GEL_BOTTOM_BC:={str(first("gel_bottom_bc", "soft"))}}}"')
print(f': "${{GEL_CONSTRAINT_STRENGTH:={float(first("gel_constraint_strength", 100.0))}}}"')
print(f': "${{INDENTOR_CONSTRAINT_STRENGTH:={float(first("indentor_constraint_strength", 100.0))}}}"')
print(f': "${{GEL_XY:={0.020}}}"')
print(f': "${{GEL_Z:={0.003}}}"')
print(f': "${{R_MIN_PROD:={float(params[:,3].min())}}}"')
print(f': "${{R_MAX_PROD:={float(params[:,3].max())}}}"')
PYEOF
)"

COMMON="--batch --indentor-geom $DRIVER_GEOM --gel-res $GEL_RES --eps-velocity $EPS_VELOCITY \
        --d-hat $D_HAT --contact-resistance $CONTACT_RESISTANCE --gel-xy $GEL_XY --gel-z $GEL_Z \
        --velocity-tol $VELOCITY_TOL --gel-bottom-bc $GEL_BOTTOM_BC \
        --gel-constraint-strength $GEL_CONSTRAINT_STRENGTH \
        --indentor-constraint-strength $INDENTOR_CONSTRAINT_STRENGTH \
        --marker-side 32 --press-steps 40 --settle-steps 10 --shear-steps 80 \
        --shear-settle 10 --batch-reps $KREPS --progress-file $PROG"

mkdir -p "$SWEEP_DIR" "$ROWS_DIR"
COMBO_META="$ROWS_DIR/_meta_${SHARD}.txt"
$PY - "$GEOM_REQ" "$NCOMBOS" "$FRAMES" "$START_COMBO" "$END_COMBO" "$ROWS_DIR" \
  "$R_MIN_PROD" "$R_MAX_PROD" > "$COMBO_META" <<'PYEOF'
import sys, os, numpy as np
geom, n, frames, start, end, rows_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
rmin, rmax = float(sys.argv[7]), float(sys.argv[8])
box = np.random.default_rng(3026)
ood_r = [0.008, 0.009, 0.010]
for ci in range(n):
    if ci < start or ci > end:
        continue
    if geom == "sphere_oodR":
        R = ood_r[ci % len(ood_r)]
    else:
        R = round(float(box.uniform(rmin, rmax)), 4)
    mu = round(float(box.uniform(0.40, 0.80)), 3)
    E = round(float(box.uniform(0.5e5, 2.0e5)), 0)
    rng = np.random.default_rng(4200 + ci)
    rows = []
    for fi in range(frames):
        depth = float(rng.uniform(0.00015, 0.00075))
        g = float(rng.uniform(0.0, 1.3))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = g * mu * 0.001
        rows.append(f"{fi} {depth:.10g} {g:.10g} {mag*np.cos(theta):.10g} {mag*np.sin(theta):.10g}")
    with open(os.path.join(rows_dir, f"combo_{ci:03d}.rows"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(ci, f"{R:.8g}", f"{mu:.8g}", f"{E:.8g}", 4200 + ci)
PYEOF

echo "GEOM-OOD SWEEP: geom=$GEOM_REQ driver_geom=$DRIVER_GEOM combos=${START_COMBO}..${END_COMBO} frames=$FRAMES K=$KREPS out=$OUT_DATA log=$PROG"
ok=0; failc=0
while read -r CI R MU E SEED; do
  combo="combo_$(printf '%03d' "$CI")"
  cdir="$SWEEP_DIR/$combo"
  rows="$ROWS_DIR/$combo.rows"
  mkdir -p "$cdir"
  echo "=== $combo: geom=$GEOM_REQ R=$R mu=$MU E=$E ==="
  cname="uipcgeom_${GEOM_REQ}_${CI}"
  docker rm -f "$cname" >/dev/null 2>&1
  timeout 7200 docker run --rm --name "$cname" --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh "$IMG" "$SCRIPT" $COMMON \
    --indentor-r="$R" --indentor-half-z="$R" --mu="$MU" --youngs="$E" --seed="$SEED" \
    --batch-rows="/work/$rows" --out="/work/$cdir"
  docker rm -f "$cname" >/dev/null 2>&1

  docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(id -u):$(id -g) /work/$cdir" >/dev/null 2>&1

  fdone=0
  for fi in $(seq 0 $((FRAMES - 1))); do
    fdir="$cdir/frame_$(printf '%03d' "$fi")"
    avg="$fdir/uipc_gt_shear_avg.npz"
    nrep="$($PY -c "import glob;print(len(glob.glob('$fdir/rep_*/uipc_gt_shear.npz')))" 2>/dev/null)"
    if [ "$nrep" = "$KREPS" ]; then
      [ -f "$avg" ] || $PY -m novbts.groundtruth.aggregate_uipc_replicates \
        --glob "$fdir/rep_*/uipc_gt_shear.npz" --out "$avg" --mode-shear-scale 0.001 >/dev/null 2>&1
      [ -f "$avg" ] && fdone=$((fdone+1))
    fi
  done
  if [ "$fdone" = "$FRAMES" ]; then echo "  $combo OK ($fdone/$FRAMES frames)"; ok=$((ok+1))
  else echo "  $combo INCOMPLETE ($fdone/$FRAMES frames)"; failc=$((failc+1)); fi
done < "$COMBO_META"

echo "GEOM-OOD SHARD ${SHARD} DONE: combos_ok=$ok incomplete=$failc"
$PY -m novbts.groundtruth.aggregate_uipc_replicates \
  --sweep-dir "$SWEEP_DIR" --out "$OUT_DATA" --mode-shear-scale 0.001 \
  --expect-reps "$KREPS" --test-size "$TEST_SIZE" --shuffle-seed 3026
