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
# Env overrides: TAG, LIVESTREAM (0/1/2), TACTILE_OVERLAY (0/1), TACTILE_OVERLAY_UPDATE_EVERY, LIVESTREAM_HOLD (s), TIMEOUT (s).
#
# With LIVESTREAM=1, connect the client to 127.0.0.1 once the progress file logs
# LIVESTREAM_VIEWPORT_CAMERA_SET (scene build takes ~4-6 min after boot):
#   tail -f sim_grasp_demo_progress.txt
#   bash ~/Downloads/kit-remote@*/kit-remote.sh
set -u
cd "$(dirname "$0")/.."

TAG="${TAG:-physics_ik_cube20_v20}"
LIVESTREAM="${LIVESTREAM:-0}"
TACTILE_OVERLAY="${TACTILE_OVERLAY:-0}"
TIMEOUT="${TIMEOUT:-2400}"

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
