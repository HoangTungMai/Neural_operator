#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WATCH_PID_FILE="${WATCH_PID_FILE:-logs/foundation_round1_resume_after_cylinder.pid}"
ROUND1_LOG="${ROUND1_LOG:-logs/foundation_round1_resume_after_cylinder.log}"
ROUND2_LOG="${ROUND2_LOG:-logs/foundation_round2_primitives.log}"
ROUND2_PID_FILE="${ROUND2_PID_FILE:-logs/foundation_round2_primitives.pid}"
POLL_SECONDS="${POLL_SECONDS:-300}"

echo "AUTO_ROUND2_START $(date -Is)"
echo "AUTO_ROUND2_WATCH_PID_FILE $WATCH_PID_FILE"

if [ ! -f "$WATCH_PID_FILE" ]; then
  echo "AUTO_ROUND2_ABORT missing_pid_file=$WATCH_PID_FILE"
  exit 2
fi

watch_pid="$(cat "$WATCH_PID_FILE")"
echo "AUTO_ROUND2_WATCH_PID $watch_pid"

while kill -0 "$watch_pid" >/dev/null 2>&1; do
  sleep "$POLL_SECONDS"
done

echo "AUTO_ROUND2_ROUND1_EXITED $(date -Is)"

if ! grep -q "FOUNDATION_ROUND1_RESUME_AFTER_CYLINDER_DONE" "$ROUND1_LOG"; then
  echo "AUTO_ROUND2_ABORT round1_done_marker_missing log=$ROUND1_LOG"
  exit 3
fi

if grep -E "INCOMPLETE|FATAL|GEOM-OOD SHARD .*incomplete=[1-9]" "$ROUND1_LOG"; then
  echo "AUTO_ROUND2_ABORT round1_log_has_failure_marker log=$ROUND1_LOG"
  exit 4
fi

echo "AUTO_ROUND2_LAUNCH_ROUND2 $(date -Is)"
setsid -f bash -lc "cd '$PWD'; echo \\\$\\\$ > '$ROUND2_PID_FILE'; exec bash infra/run_foundation_round2_primitives.sh" > "$ROUND2_LOG" 2>&1
echo "AUTO_ROUND2_ROUND2_PID_FILE $ROUND2_PID_FILE"
echo "AUTO_ROUND2_DONE $(date -Is)"
