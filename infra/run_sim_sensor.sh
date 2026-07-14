#!/usr/bin/env bash
# Standalone one-pad VBTS runner.  Script mode is host-only; manual mode starts
# the existing Isaac Docker plumbing through run_sim_grasp.sh's sensor branch.
set -u
cd "$(dirname "$0")/.."

PY="${PY:-.venv-gate2/bin/python}"
DRIVE="${DRIVE:-script}"
LIVESTREAM="${LIVESTREAM:-0}"
TAG="${TAG:-sensor_script}"
AUTO_CONNECT="${AUTO_CONNECT:-1}"
TIMEOUT="${TIMEOUT:-3600}"
PROGRESS="sim_sensor_progress.txt"

gate_sensor_npz() {
  "$PY" - "$1" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
z = np.load(path, allow_pickle=False)
required = {"images", "pen_max", "shear", "pose", "flow", "phase", "timings_s"}
missing = required.difference(z.files)
if missing:
    raise SystemExit(f"missing keys: {sorted(missing)}")
images, pen_max, shear, phase = z["images"], z["pen_max"], z["shear"], z["phase"]
if images.ndim != 3 or images.shape[1:] != (160, 160):
    raise SystemExit(f"unexpected images shape {images.shape}")
if pen_max.shape != (images.shape[0],) or shear.shape != (images.shape[0], 2):
    raise SystemExit("inconsistent sequence lengths")
press = pen_max[phase == "press"]
if not (pen_max.max() > 0.0005 and np.all(np.diff(press) >= -1e-9)):
    raise SystemExit("penetration gate failed")
rest = images[np.flatnonzero(phase == "rest")[0]]
pressed = images[np.flatnonzero(phase == "press")[-1]]
delta = np.abs(pressed - rest)
if not (delta.mean() > 0.001 and delta.max() > 0.05 and rest.std() > 0.03 and pressed.std() > 0.03):
    raise SystemExit(
        f"image gate failed mean={delta.mean():.6f} max={delta.max():.6f} "
        f"std=({rest.std():.6f},{pressed.std():.6f})"
    )
print("sensor_npz_ok", path, "frames", images.shape[0], "mae", float(delta.mean()))
PY
}

autoconnect_pid=""
start_autoconnect() {
  local dir
  dir="${KIT_REMOTE:-$(ls -d "$HOME"/Downloads/kit-remote@*/ 2>/dev/null | head -1)}"
  if [ -z "$dir" ] || [ ! -f "$dir/kit-remote.sh" ]; then
    echo "[run_sim_sensor] no kit-remote client; set KIT_REMOTE or AUTO_CONNECT=0" >&2
    return
  fi
  mkdir -p logs
  if pgrep -f omniverse-streaming-client >/dev/null 2>&1; then
    pkill -f 'kit-remote.sh' 2>/dev/null || true
    pkill -f omniverse-streaming-client 2>/dev/null || true
  fi
  (
    deadline=$((SECONDS + TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
      if grep -q LIVESTREAM_VIEWPORT_CAMERA_SET "$PROGRESS" 2>/dev/null; then
        disp="${DISPLAY:-}"
        if [ -z "$disp" ]; then
          sock=$(ls /tmp/.X11-unix/X* 2>/dev/null | head -1)
          disp=":${sock##*/X}"
        fi
        exec env DISPLAY="$disp" bash "$dir/kit-remote.sh" -s "${STREAM_SERVER:-127.0.0.1}" \
          >"logs/kit_remote_${TAG}.log" 2>&1
      fi
      sleep 2
    done
  ) &
  autoconnect_pid=$!
}

stop_autoconnect() {
  [ -n "$autoconnect_pid" ] && kill "$autoconnect_pid" 2>/dev/null || true
}
on_signal() {
  stop_autoconnect
  exit 130
}
trap stop_autoconnect EXIT
trap on_signal INT TERM

rm -f "$PROGRESS"
if [ "$DRIVE" = "script" ] && [ "$LIVESTREAM" = "0" ]; then
  effective_tag="$TAG"
  custom_out=""
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "--tag" ]; then
      effective_tag="$arg"
    fi
    if [ "$prev" = "--out" ]; then
      custom_out="$arg"
    fi
    prev="$arg"
  done
  SENSOR_PROGRESS="$PWD/$PROGRESS" PYTHONPATH=src "$PY" -m novbts.simulation.runtime.sensor_demo \
    --runtime host --drive script --tag "$TAG" --no-noise "$@" || exit $?
  out="${custom_out:-runs/sim_sensor/$effective_tag}"
  grep -q SENSOR_DEMO_OK "$PROGRESS" || exit 1
  gate_sensor_npz "$out/sensor_sequence.npz" || exit 1
  for png in tactile_rest.png tactile_pressed.png tactile_shear.png tactile_released.png; do
    test -s "$out/$png" || exit 1
  done
  exit 0
fi

if [ "$DRIVE" != "manual" ]; then
  echo "[run_sim_sensor] Isaac runtime supports DRIVE=manual only; use LIVESTREAM=0 for script gates." >&2
  exit 2
fi
if [ "$LIVESTREAM" = "0" ]; then
  echo "[run_sim_sensor] manual mode needs LIVESTREAM=1 or LIVESTREAM=2." >&2
  exit 2
fi
if [ "$LIVESTREAM" = "1" ] && [ "$AUTO_CONNECT" = "1" ]; then
  start_autoconnect
fi
SENSOR_PROGRESS="$PWD/$PROGRESS" TIMEOUT="$TIMEOUT" \
  bash infra/run_sim_grasp.sh sensor --drive manual --livestream "$LIVESTREAM" --tag "$TAG" "$@"
