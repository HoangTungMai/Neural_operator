#!/usr/bin/env bash
# Canonical P4b-passing grasp: UR3e + Robotiq 2F-85 + VBTS lifts a 20 mm cube.
#
# This is the V19/V20 winning IK command (object_lift_m ~= 0.119, max_force ~= 20.8 N,
# p4b_gate_pass=true). The gains below are load-bearing -- see codex/REPORT_isaac_grasp_env.md
# and memory isaac-grasp-rootcause-v14 (ride-up offset + low lift/hold orientation weight).
#
# Usage:
#   bash infra/run_grasp_pick_v20.sh                  # headless, data only
#   LIVESTREAM=1 bash infra/run_grasp_pick_v20.sh     # + native Omniverse Streaming Client
#
# Env overrides: TAG, LIVESTREAM (0/1/2), TACTILE_OVERLAY (0/1), TACTILE_OVERLAY_UPDATE_EVERY,
# LIVESTREAM_HOLD (s), TIMEOUT (s), AUTO_CONNECT (0/1), KIT_REMOTE (dir), STREAM_SERVER (ip).
#
# With LIVESTREAM=1 the client is launched automatically once the progress file logs
# LIVESTREAM_VIEWPORT_CAMERA_SET (scene build takes ~4-6 min after boot). Connecting before
# that marker gives a black screen. Set AUTO_CONNECT=0 to launch it yourself instead:
#   tail -f sim_grasp_demo_progress.txt
#   bash ~/Downloads/kit-remote@*/kit-remote.sh -s 127.0.0.1
set -u
cd "$(dirname "$0")/.."

TAG="${TAG:-physics_ik_cube20_v20}"
LIVESTREAM="${LIVESTREAM:-0}"
TACTILE_OVERLAY="${TACTILE_OVERLAY:-0}"
TIMEOUT="${TIMEOUT:-2400}"
AUTO_CONNECT="${AUTO_CONNECT:-1}"
STREAM_SERVER="${STREAM_SERVER:-127.0.0.1}"
PROGRESS=sim_grasp_demo_progress.txt

# Native livestream only (mode 2 is browser/WebRTC, nothing to launch).
autoconnect_pid=""
start_autoconnect() {
  local dir
  dir="${KIT_REMOTE:-$(ls -d "$HOME"/Downloads/kit-remote@*/ 2>/dev/null | head -1)}"
  if [ -z "$dir" ] || [ ! -f "$dir/kit-remote.sh" ]; then
    echo "[run_grasp_pick_v20] no kit-remote client found; set KIT_REMOTE=<dir> or AUTO_CONNECT=0" >&2
    return
  fi
  mkdir -p logs
  # A client left over from an earlier run keeps its window open pointing at a dead
  # server, and the two windows are indistinguishable. Retire it before opening ours.
  if pgrep -f omniverse-streaming-client >/dev/null 2>&1; then
    echo "[run_grasp_pick_v20] closing stale streaming client(s)" >&2
    pkill -f 'kit-remote.sh' 2>/dev/null
    pkill -f omniverse-streaming-client 2>/dev/null
  fi
  (
    # The marker is written by grasp_livestream.py once the viewport camera is framed.
    deadline=$(( SECONDS + TIMEOUT ))
    while [ "$SECONDS" -lt "$deadline" ]; do
      if grep -q LIVESTREAM_VIEWPORT_CAMERA_SET "$PROGRESS" 2>/dev/null; then
        echo "[run_grasp_pick_v20] viewport ready, connecting client to $STREAM_SERVER" >&2
        # Fall back to whatever X socket exists; this host runs :1, not :0.
        disp="${DISPLAY:-}"
        if [ -z "$disp" ]; then
          sock=$(ls /tmp/.X11-unix/X* 2>/dev/null | head -1)
          disp=":${sock##*/X}"
        fi
        exec env DISPLAY="$disp" bash "$dir/kit-remote.sh" -s "$STREAM_SERVER" \
          >logs/kit_remote_"$TAG".log 2>&1
      fi
      sleep 2
    done
    echo "[run_grasp_pick_v20] gave up waiting for LIVESTREAM_VIEWPORT_CAMERA_SET" >&2
  ) &
  autoconnect_pid=$!
}
stop_autoconnect() {
  [ -n "$autoconnect_pid" ] && kill "$autoconnect_pid" 2>/dev/null
  return 0
}
on_signal() {
  stop_autoconnect
  exit 130
}
trap stop_autoconnect EXIT
trap on_signal INT TERM

stream_env=()
if [ "$LIVESTREAM" != "0" ]; then
  stream_env=(LIVESTREAM_ON=1 LIVESTREAM_MODE="$LIVESTREAM" LIVESTREAM_HOLD="${LIVESTREAM_HOLD:-600}")
fi
overlay_args=()
if [ "$TACTILE_OVERLAY" = "1" ]; then
  # Inline VBTS rendering is opt-in; cadence 8 keeps the LIFT slowdown under 15%.
  overlay_args=(--tactile-overlay --tactile-overlay-update-every "${TACTILE_OVERLAY_UPDATE_EVERY:-8}")
fi

rm -f sim_grasp_demo_progress.txt

if [ "$LIVESTREAM" = "1" ] && [ "$AUTO_CONNECT" = "1" ]; then
  start_autoconnect
fi

env "${stream_env[@]}" TIMEOUT="$TIMEOUT" \
  bash infra/run_sim_grasp.sh demo pick --tag "$TAG" \
  --object cube --object-size 0.020 --robot-z 0.35 --live-pose-tactile \
  --physics-lift-probe --no-noise --grasp-mode ik --ik-solver weighted-dls \
  --ik-orientation-weight 0.30 --ik-lift-orientation-weight 0.03 \
  --ik-hold-orientation-weight 0.03 --ik-z-priority-error-m 0 \
  --ik-position-weight 1.0 --ik-z-position-weight 4.0 --ik-control-substeps 4 \
  --ik-max-joint-step 0.12 --ik-pedestal-height 0.35 --ik-min-close-ee-z 0.36 \
  --ik-grip-axis-world "1,0,0" --ik-soft-close-steps 600 \
  --ik-close-gripper-stiffness 30 --ik-close-gripper-damping 6 \
  --ik-close-gripper-effort-limit 2.0 --mu 1.5 --ik-place-at-selected-close-target \
  --ik-place-ride-up-offset 0.008 --physics-lift-height 0.12 \
  --physics-lift-steps 960 --ik-lift-gripper-stiffness 45 \
  --ik-lift-gripper-damping 8 --ik-lift-gripper-effort-limit 4 \
  --ik-hold-gripper-stiffness 45 --ik-hold-gripper-damping 8 \
  --ik-hold-gripper-effort-limit 4 --ik-lift-live-midpoint-servo \
  --ik-lift-gripper-target-offset 0.002 --ik-hold-gripper-target-offset 0.002 \
  --no-physics-require-force-gate --world-camera "${overlay_args[@]}"
