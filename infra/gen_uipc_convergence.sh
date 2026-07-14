#!/usr/bin/env bash
# UIPC (TacEx GIPC) convergence sweep for ONE indent+shear configuration.
#
# PURPOSE: answer "does the IPC tangential field CONVERGE?" — the question PhysX
# fails (paired tangential rel-L2 wanders 0.7-0.9 under mesh refinement instead of
# shrinking; see memory res32-upgrade). IPC should converge along BOTH knobs:
#   * edge_length_r -> 0   (mesh refinement; barrier contact is convergent)
#   * eps_velocity  -> 0   (smooth-friction model limit; mesh-independent)
#
# We run TWO 1-D refinement PATHS (not the full grid) — exactly what the
# successive-level rel-L2 analysis needs, at ~8 runs instead of 20:
#   mesh path : eps_velocity fixed at the FINEST value, gel_res swept (finer = larger)
#   fric path : gel_res fixed at the FINEST value, eps_velocity swept (finer = smaller)
# The finest corner is shared by both paths.
#
# gel_res = a DETERMINISTIC structured-mesh resolution (NOT wildmeshing's
# edge_length_r): the previous wildmeshing sweep was invalid because the mesh
# changed every run and the irregular top face collapsed the sampled field.
#
# Each (edge_length_r, eps_velocity) runs in its OWN container/process via the
# driver's --single mode (à la gen_fem_sweep*.sh). This SIDESTEPS the unverified
# in-process multi-UipcSim path: every process gets a clean engine/SimulationContext.
# Aggregate afterwards with:
#   python -m novbts.research.groundtruth.aggregate_uipc_convergence --conv-dir data/uipc/conv
#
# Config is the realistic thin-gel Phase-0 config (gel 0.020/0.003, R 0.004,
# mu 0.6, E 1e5, depth 0.00045, shear 0.00036). NOTE: this is IPC self-consistency, NOT yet a paired
# comparison vs PhysX — that is a separate run matching a PhysX combo's geometry.
#
# Usage: bash infra/gen_uipc_convergence.sh
# RESUMABLE: a setting whose npz already exists is skipped.
set -u
cd "$(dirname "$0")/.."
IMG=isaac-lab-tacex:latest
SCRIPT=/work/src/novbts/research/groundtruth/tacex_uipc_extract_shear.py
NAME=uipcconv
PY=.venv-gate2/bin/python
GEL_XY="${GEL_XY:-0.020}"
GEL_Z="${GEL_Z:-0.003}"
INDENTOR_R="${INDENTOR_R:-0.004}"
MU="${MU:-0.6}"
YOUNGS="${YOUNGS:-1.0e5}"
DEPTH="${DEPTH:-0.00030}"
SHEAR="${SHEAR:-0.00036}"
DRIVE_RATIO="${DRIVE_RATIO:-0.60}"
D_HAT="${D_HAT:-0.0001}"
CONTACT_RESISTANCE="${CONTACT_RESISTANCE:-1.0e9}"
VELOCITY_TOL="${VELOCITY_TOL:-0.001}"
GEL_BOTTOM_BC="${GEL_BOTTOM_BC:-soft}"
GEL_CONSTRAINT_STRENGTH="${GEL_CONSTRAINT_STRENGTH:-100}"
INDENTOR_CONSTRAINT_STRENGTH="${INDENTOR_CONSTRAINT_STRENGTH:-100}"
RES_LEVELS_STR="${RES_LEVELS_STR:-16 20 24}"
EPS_LEVELS_STR="${EPS_LEVELS_STR:-0.001 0.0005 0.00025 0.0001 0.00005 0.000025}"
CONV_DIR="${CONV_DIR:-data/uipc/conv_realistic}"
GEL_NZ="${GEL_NZ:-}"

GEL_NZ_ARG=()
if [ -n "$GEL_NZ" ]; then
  GEL_NZ_ARG=(--gel-nz "$GEL_NZ")
fi

# fixed indent+shear config (the smoke config) — shared by every run
COMMON="--single --marker-side 32 --gel-xy $GEL_XY --gel-z $GEL_Z \
  --indentor-r $INDENTOR_R --mu $MU --youngs $YOUNGS --depth $DEPTH --shear $SHEAR \
  --drive-ratio $DRIVE_RATIO --d-hat $D_HAT --contact-resistance $CONTACT_RESISTANCE \
  --velocity-tol $VELOCITY_TOL --gel-bottom-bc $GEL_BOTTOM_BC \
  --gel-constraint-strength $GEL_CONSTRAINT_STRENGTH \
  --indentor-constraint-strength $INDENTOR_CONSTRAINT_STRENGTH \
  --press-steps 40 --settle-steps 10 --shear-steps 80 --shear-settle 10"

verify_config() {
  "$PY" -c '
import sys
import numpy as np

path, res, eps, vtol, bottom_bc, gel_strength, indentor_strength, gel_nz = sys.argv[1:]
with np.load(path, allow_pickle=True) as data:
    def scalar(key):
        if key not in data.files:
            raise AssertionError(f"{path}: missing provenance key {key}")
        return np.asarray(data[key]).reshape(-1)[0]

    expected = {
        "gel_res": int(res),
        "eps_velocity": float(eps),
        "velocity_tol": float(vtol),
        "gel_bottom_bc": bottom_bc,
        "gel_constraint_strength": float(gel_strength),
        "indentor_constraint_strength": float(indentor_strength),
    }
    for key, want in expected.items():
        got = scalar(key)
        if isinstance(want, str):
            if str(got) != want:
                raise AssertionError(f"{path}: {key}={got!r}, expected {want!r}")
        elif not np.isclose(float(got), want, rtol=1e-6, atol=1e-12):
            raise AssertionError(f"{path}: {key}={got!r}, expected {want!r}")
    if gel_nz:
        got_nz = scalar("gel_nz")
        if int(got_nz) != int(gel_nz):
            raise AssertionError(
                f"{path}: gel_nz={got_nz!r}, expected {gel_nz!r}"
            )
    disp = np.asarray(data["disp"])
    if not np.isfinite(disp).all():
        raise AssertionError(f"{path}: disp contains non-finite values")
' "$@"
}

# refinement levels (finest LAST). Edit here to extend/shorten the study.
# mesh: gel_res cells/footprint-axis (finer = LARGER). fric: eps_velocity (finer = smaller).
read -r -a RES_LEVELS <<< "$RES_LEVELS_STR"
read -r -a EPS_LEVELS <<< "$EPS_LEVELS_STR"
RES_FINE="${RES_LEVELS[${#RES_LEVELS[@]}-1]}"   # 24
EPS_FINE="${EPS_LEVELS[${#EPS_LEVELS[@]}-1]}"   # 0.001

mkdir -p "$CONV_DIR"

# Build the unique (gel_res, eps_velocity) point list for the two paths.
# mesh path: every RES level at EPS_FINE ; fric path: RES_FINE at every EPS level.
declare -A SEEN
POINTS=()
for r in "${RES_LEVELS[@]}"; do POINTS+=("$r $EPS_FINE"); SEEN["$r $EPS_FINE"]=1; done
for ev in "${EPS_LEVELS[@]}"; do
  key="$RES_FINE $ev"
  if [ -z "${SEEN[$key]:-}" ]; then POINTS+=("$key"); SEEN[$key]=1; fi
done

# tag helper: res12_ev0.005
tag() { printf 'res%s_ev%s' "$1" "$2"; }

ndone=0; nrun=0; nfail=0
echo "UIPC convergence: ${#POINTS[@]} unique settings (mesh path ${#RES_LEVELS[@]} + fric path ${#EPS_LEVELS[@]} - 1 shared)"
echo "config: gel_bottom_bc=$GEL_BOTTOM_BC gel_strength=$GEL_CONSTRAINT_STRENGTH indentor_strength=$INDENTOR_CONSTRAINT_STRENGTH velocity_tol=$VELOCITY_TOL gel_nz=${GEL_NZ:-auto}"
for pt in "${POINTS[@]}"; do
  read RES EV <<< "$pt"
  t="$(tag "$RES" "$EV")"
  out="/work/$CONV_DIR/$t"
  host="$CONV_DIR/$t/uipc_gt_shear.npz"
  if [ -f "$host" ]; then
    if verify_config "$host" "$RES" "$EV" "$VELOCITY_TOL" "$GEL_BOTTOM_BC" \
        "$GEL_CONSTRAINT_STRENGTH" "$INDENTOR_CONSTRAINT_STRENGTH" "$GEL_NZ"; then
      echo "skip $t (npz exists; provenance verified)"; ndone=$((ndone+1)); continue
    fi
    echo "$t FAILED (existing npz provenance mismatch; choose a fresh CONV_DIR)"
    nfail=$((nfail+1))
    continue
  fi
  echo "=== $t : gel_res=$RES eps_velocity=$EV ==="
  docker rm -f $NAME >/dev/null 2>&1
  timeout 1200 docker run --rm --name $NAME --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh $IMG $SCRIPT $COMMON \
    --gel-res "$RES" --eps-velocity "$EV" --out "$out" "${GEL_NZ_ARG[@]}"
  docker rm -f $NAME >/dev/null 2>&1
  docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(id -u):$(id -g) /work/$CONV_DIR" >/dev/null 2>&1
  # Container exit code is unreliable; gate success on the npz existing + loadable.
  if [ -f "$host" ] && verify_config "$host" "$RES" "$EV" "$VELOCITY_TOL" \
      "$GEL_BOTTOM_BC" "$GEL_CONSTRAINT_STRENGTH" \
      "$INDENTOR_CONSTRAINT_STRENGTH" "$GEL_NZ"; then
    echo "$t OK"; nrun=$((nrun+1))
  else
    echo "$t FAILED (no valid npz)"; nfail=$((nfail+1))
  fi
done
echo "UIPC CONVERGENCE DONE: skipped=$ndone ran=$nrun failed=$nfail / ${#POINTS[@]} settings"
if [ "$nfail" -ne 0 ]; then
  exit 1
fi
echo "Aggregate: $PY -m novbts.research.groundtruth.aggregate_uipc_convergence --conv-dir $CONV_DIR"
