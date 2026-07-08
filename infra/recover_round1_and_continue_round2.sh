#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

POLL_SECONDS="${POLL_SECONDS:-900}"
STALE_SECONDS="${STALE_SECONDS:-3600}"
PY="${PY:-.venv-gate2/bin/python}"

bolt_dir="data/uipc/geom_ood/mesh/bolt_hex/sweep/combo_011"
bolt_progress="fem_progress_uipc_geom_bolt_hex_6_11.txt"

count_raw() {
  find "$bolt_dir" -name uipc_gt_shear.npz -type f 2>/dev/null | wc -l
}

progress_mtime() {
  stat -c %Y "$bolt_progress" 2>/dev/null || echo 0
}

echo "RECOVER_ROUND1_CONTINUE_START $(date -Is)"
echo "RECOVER_WAIT_BOLT_HEX_COMBO_011 poll_seconds=$POLL_SECONDS stale_seconds=$STALE_SECONDS"

while true; do
  raw="$(count_raw)"
  echo "RECOVER_BOLT_HEX_COMBO_011_STATUS $(date -Is) raw=$raw/75"
  if [ "$raw" -ge 75 ]; then
    break
  fi

  now="$(date +%s)"
  mtime="$(progress_mtime)"
  age=$((now - mtime))
  if [ "$raw" -lt 75 ] && [ "$age" -ge "$STALE_SECONDS" ]; then
    echo "RECOVER_BOLT_HEX_COMBO_011_STALE $(date -Is) age=$age raw=$raw; rerun combo_011"
    IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh bolt_hex 12 25 3 11 11
    raw="$(count_raw)"
    if [ "$raw" -lt 75 ]; then
      echo "RECOVER_ABORT bolt_hex_combo_011_incomplete_after_rerun raw=$raw"
      exit 5
    fi
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "RECOVER_AGGREGATE_BOLT_HEX $(date -Is)"
"$PY" -m novbts.groundtruth.aggregate_uipc_replicates \
  --sweep-dir data/uipc/geom_ood/mesh/bolt_hex/sweep \
  --out data/uipc/geom_ood/mesh/bolt_hex/bolt_hex_avg.npz \
  --mode-shear-scale 0.001 --expect-reps 3 --test-size 100 --shuffle-seed 3026

echo "RECOVER_ROUND1_REMAINING_START $(date -Is)"
IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh rounded_tip 12 25 3 6 11

for g in cone multi_lobe rounded_die gear pebble; do
  echo "RECOVER_ROUND1_NEW_START $g $(date -Is)"
  IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh "$g" 12 25 3 0 11
  echo "RECOVER_ROUND1_NEW_DONE $g $(date -Is)"
done

echo "RECOVER_ROUND1_DONE $(date -Is)"
echo "RECOVER_ROUND2_START $(date -Is)"
bash infra/run_foundation_round2_primitives.sh
echo "RECOVER_ROUND1_AND_ROUND2_DONE $(date -Is)"
