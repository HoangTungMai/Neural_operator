#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

POLL_SECONDS="${POLL_SECONDS:-900}"
LOG_OUT="${LOG_OUT:-logs/foundation_progress_monitor.log}"

mkdir -p logs

count_npz() {
  local dir="$1"
  local name="$2"
  if [ -d "$dir" ]; then
    find "$dir" -name "$name" -type f 2>/dev/null | wc -l
  else
    echo 0
  fi
}

shape_base() {
  case "$1" in
    bolt_hex|rounded_tip|cone|multi_lobe|rounded_die|gear|pebble)
      echo "data/uipc/geom_ood/mesh/$1"
      ;;
    *)
      echo "data/uipc/geom_ood/$1"
      ;;
  esac
}

report_shape() {
  local g="$1"
  local start="$2"
  local end="$3"
  local base
  base="$(shape_base "$g")"
  printf "%s" "$g"
  for ci in $(seq "$start" "$end"); do
    local combo
    combo="$(printf "%03d" "$ci")"
    local d="$base/sweep/combo_$combo"
    local raw avg
    raw="$(count_npz "$d" uipc_gt_shear.npz)"
    avg="$(count_npz "$d" uipc_gt_shear_avg.npz)"
    printf " c%s=%s/%s" "$combo" "$raw" "$avg"
  done
  printf "\n"
}

while true; do
  {
    echo "===== FOUNDATION_MONITOR $(date -Is) ====="
    echo "-- active processes --"
    ps -ef | grep -E "recover_round|run_foundation|gen_uipc|docker run|uipcgeom|python.sh" | grep -v grep || true
    echo "-- docker --"
    docker ps --format "{{.Names}} {{.Status}}" 2>&1 || true
    echo "-- latest progress files --"
    ls -lt fem_progress_uipc_geom_*_*.txt 2>/dev/null | head -12 || true
    echo "-- round1 top-up counts --"
    report_shape sphere_oodR 6 11
    report_shape cuboid 6 11
    report_shape ellipsoid 6 11
    report_shape bolt_hex 6 11
    report_shape rounded_tip 6 11
    echo "-- round1 new/held-out counts --"
    report_shape cone 0 11
    report_shape multi_lobe 0 11
    report_shape rounded_die 0 11
    report_shape gear 0 11
    report_shape pebble 0 11
    echo "-- round2 primitive counts --"
    report_shape cylinder 12 23
    report_shape sphere_oodR 12 23
    report_shape cuboid 12 23
    report_shape ellipsoid 12 23
    report_shape bolt_hex 12 23
    report_shape rounded_tip 12 23
    report_shape cone 12 23
    report_shape multi_lobe 12 23
    echo
  } >> "$LOG_OUT" 2>&1
  sleep "$POLL_SECONDS"
done
