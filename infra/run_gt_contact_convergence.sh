#!/usr/bin/env bash
# Production-matched, provenance-gated UIPC contact convergence sweep.
# Run this script under setsid/nohup; it is resumable and never overwrites a
# mismatched or unreadable result.
set -euo pipefail
cd "${BASH_SOURCE[0]%/*}/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
SCRIPT=/work/src/novbts/research/groundtruth/tacex_uipc_extract_shear.py
PY="${PY:-.venv-gate2/bin/python}"
SOURCE="${SOURCE:-data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz}"
OUT_ROOT="${OUT_ROOT:-data/uipc/gt_contact_convergence_codex}"
CONFIGS="${CONFIGS:-codex/gt_contact_axis_configs.tsv}"
ROWS="$OUT_ROOT/frame_rows.txt"
META="$OUT_ROOT/frame_meta.tsv"
TIMING="$OUT_ROOT/container_wall_times.tsv"
LOG="$OUT_ROOT/runner.log"
CONTAINER="uipcgtcc"

rtk proxy mkdir -p "$OUT_ROOT"
rtk proxy "$PY" infra/analyze_gt_contact_convergence.py prepare \
  --source "$SOURCE" --rows "$ROWS" --meta "$META"

if [ ! -f "$TIMING" ]; then
  printf 'tag\tcontainer_wall_s\n' > "$TIMING"
fi

while IFS=$'\t' read -r TAG GEL_RES SUBDIV D_HAT EPS RESISTANCE PRESS SETTLE SHEAR SHEAR_SETTLE DT; do
  if [ "$TAG" = "tag" ] || [ -z "$TAG" ]; then
    continue
  fi
  if rtk proxy "$PY" infra/analyze_gt_contact_convergence.py verify \
      --root "$OUT_ROOT" --configs "$CONFIGS" --rows "$ROWS" --tag "$TAG" >/dev/null 2>&1; then
    echo "skip $TAG (3 NPZ files exist and provenance matches)"
    continue
  fi
  if rtk proxy find "$OUT_ROOT/$TAG" -name uipc_gt_shear.npz -print -quit 2>/dev/null | rtk proxy grep -q .; then
    echo "FAILED $TAG: partial or provenance-mismatched output exists; use a fresh OUT_ROOT" >&2
    exit 1
  fi

  echo "=== $TAG res=$GEL_RES subdiv=$SUBDIV dhat=$D_HAT eps=$EPS resistance=$RESISTANCE shear_steps=$SHEAR dt=$DT ==="
  rtk docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  start=$SECONDS
  rtk docker run --rm --name "$CONTAINER" --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 \
    -v "$PWD":/work --entrypoint /isaac-sim/python.sh "$IMG" "$SCRIPT" \
    --batch --batch-rows "/work/$ROWS" --batch-reps 1 \
    --out "/work/$OUT_ROOT/$TAG" --progress-file "/work/$OUT_ROOT/progress_$TAG.log" \
    --gel-res "$GEL_RES" --gel-nz 24 --gel-xy 0.020 --gel-z 0.003 \
    --indentor-subdiv "$SUBDIV" --eps-velocity "$EPS" --d-hat "$D_HAT" \
    --contact-resistance "$RESISTANCE" --velocity-tol 1e-5 \
    --gel-bottom-bc fixed --gel-constraint-strength 0 \
    --indentor-constraint-strength 30000 --marker-side 32 \
    --press-steps "$PRESS" --settle-steps "$SETTLE" \
    --shear-steps "$SHEAR" --shear-settle "$SHEAR_SETTLE" --dt "$DT" \
    --seed 1362
  rtk docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  rtk docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(rtk proxy id -u):$(rtk proxy id -g) /work/$OUT_ROOT" >/dev/null
  wall=$((SECONDS - start))
  printf '%s\t%s\n' "$TAG" "$wall" >> "$TIMING"
  rtk proxy "$PY" infra/analyze_gt_contact_convergence.py verify \
    --root "$OUT_ROOT" --configs "$CONFIGS" --rows "$ROWS" --tag "$TAG"
done < "$CONFIGS"

rtk proxy "$PY" infra/analyze_gt_contact_convergence.py summarize \
  --root "$OUT_ROOT" --configs "$CONFIGS" --rows "$ROWS" \
  --output "$OUT_ROOT/metrics.json"
rtk proxy echo "GT CONTACT CONVERGENCE SWEEP COMPLETE -> $OUT_ROOT" | rtk proxy tee -a "$LOG"
