#!/usr/bin/env bash
# Isaac Sim UR3e/Robotiq/VBTS simulation runner.
set -u
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
SUB="${1:-}"
shift || true

# Persist Isaac's Kit/RTX/compute caches across `docker run --rm` (the image ships
# them empty). Measured on this box (boot = launch -> first progress line):
#   rendering run (--livestream/--world-camera): cold 261 s -> warm 109 s, cache 399 MB
#   headless press:                              cold  22 s -> warm  23 s, cache 2.4 MB
# So the win is entirely on rendering runs; headless boots are already cheap.
# Keep the store on the NVMe home, NOT under the HDD-backed docker data-root.
# NO_ISAAC_CACHE=1 opts out. The container writes these as root: wipe them with
#   docker run --rm -v "$HOME/.cache":/hc --entrypoint bash "$IMG" -c "rm -rf /hc/isaac-sim-docker"
ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/.cache/isaac-sim-docker}"
CACHE_MOUNTS=()
if [ "${NO_ISAAC_CACHE:-0}" != "1" ]; then
  mkdir -p "$ISAAC_CACHE_DIR"/{ov,kit,computecache,glcache,localov,warp}
  CACHE_MOUNTS=(
    -v "$ISAAC_CACHE_DIR/ov":/root/.cache/ov
    -v "$ISAAC_CACHE_DIR/kit":/isaac-sim/kit/cache
    -v "$ISAAC_CACHE_DIR/computecache":/root/.nv/ComputeCache
    -v "$ISAAC_CACHE_DIR/glcache":/root/.cache/nvidia/GLCache
    -v "$ISAAC_CACHE_DIR/localov":/root/.local/share/ov
    -v "$ISAAC_CACHE_DIR/warp":/root/.cache/warp
  )
fi

DOCKER_BASE=(docker run --rm --gpus all
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0
  -v "$PWD":/work
  "${CACHE_MOUNTS[@]}"
  --entrypoint /isaac-sim/python.sh "$IMG")

chown_back() {
  docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(id -u):$(id -g) /work/data/assets /work/runs/sim_grasp /work/*_progress.txt 2>/dev/null || true" >/dev/null 2>&1 || true
}

gate_marker() {
  local marker="$1"; local prog="$2"
  if [ ! -f "$prog" ]; then
    echo "missing progress file: $prog" >&2
    return 1
  fi
  if ! grep -q "$marker" "$prog"; then
    echo "missing marker '$marker' in $prog" >&2
    tail -80 "$prog" >&2 || true
    return 1
  fi
}

gate_json() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import json, sys
p=sys.argv[1]
with open(p) as f: json.load(f)
print("json_ok", p)
PY
}

gate_npz() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import numpy as np, sys
p=sys.argv[1]
z=np.load(p, allow_pickle=True)
print("npz_ok", p, z.files)
PY
}

case "$SUB" in
  check)
    rm -f sim_grasp_asset_progress.txt
    timeout "${TIMEOUT:-600}" "${DOCKER_BASE[@]}" /work/src/novbts/sim/build_robot_asset.py --check-only "$@"
    chown_back
    gate_marker CHECK_OK sim_grasp_asset_progress.txt || exit 1
    gate_json data/assets/asset_check.json || exit 1
    ;;
  build-asset)
    rm -f sim_grasp_asset_progress.txt
    timeout "${TIMEOUT:-1200}" "${DOCKER_BASE[@]}" /work/src/novbts/sim/build_robot_asset.py "$@"
    chown_back
    gate_marker ASSET_BUILD_OK sim_grasp_asset_progress.txt || exit 1
    gate_json data/assets/ur3e_robotiq_vbts.json || exit 1
    test -s data/assets/ur3e_robotiq_vbts.usd || exit 1
    ;;
  verify-asset)
    rm -f sim_grasp_asset_progress.txt
    timeout "${TIMEOUT:-900}" "${DOCKER_BASE[@]}" /work/src/novbts/sim/build_robot_asset.py --verify "$@"
    chown_back
    gate_marker ASSET_VERIFY_OK sim_grasp_asset_progress.txt || exit 1
    gate_json data/assets/asset_verify.json || exit 1
    ;;
  demo)
    mode="${1:-press}"
    if [ "$#" -gt 0 ]; then shift; fi
    tag=""
    prev=""
    livestream_requested="${LIVESTREAM_ON:-0}"
    livestream_hold="${LIVESTREAM_HOLD:-60}"
    # LIVESTREAM_MODE follows the IsaacLab/IsaacSim image in use. Recent images use
    # 2 for local/private WebRTC; older IsaacSim 4.x images warn that 1 is native.
    livestream_mode="${LIVESTREAM_MODE:-2}"
    # WebRTC streaming clients need the container to see the host interface; keep
    # bridge+8211 available for older native/browser builds.
    livestream_network="${LIVESTREAM_NETWORK:-host}"
    public_ip="${PUBLIC_IP:-127.0.0.1}"
    for arg in "$@"; do
      if [ "$prev" = "--tag" ]; then
        tag="$arg"
      fi
      if [ "$prev" = "--livestream" ] && [ "$arg" != "0" ]; then
        livestream_requested=1
      fi
      if [ "$prev" = "--livestream" ] && [ "$arg" != "0" ]; then
        livestream_mode="$arg"
      fi
      if [ "$prev" = "--livestream-hold" ]; then
        livestream_hold="$arg"
      fi
      prev="$arg"
    done
    driver_args=("$@")
    docker_base=("${DOCKER_BASE[@]}")
    demo_timeout="${TIMEOUT:-1800}"
    if [ "$livestream_requested" = "1" ]; then
      docker_base=(docker run --rm --gpus all
        -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES
        -e LIVESTREAM="$livestream_mode" -e PUBLIC_IP="$public_ip")
      if [ "$livestream_network" = "host" ]; then
        docker_base+=(--network host)
      else
        docker_base+=(-p 8211:8211/tcp -p 8211:8211/udp -p 49100:49100/tcp -p 47998:47998/udp)
      fi
      # Native livestream (mode 1) needs a GPU windowing surface: it loads
      # carb.windowing-glfw and captures the on-screen viewport. Without a
      # display the control channel connects but the client gets zero video
      # frames -> black screen. WebRTC (mode 2) is offscreen and needs none.
      if [ "$livestream_mode" = "1" ]; then
        ls_display="${LIVESTREAM_DISPLAY:-${DISPLAY:-}}"
        if [ -z "$ls_display" ]; then
          # Pick the first existing X socket (e.g. :1 from a gdm login session).
          ls_display=":$(ls /tmp/.X11-unix/ 2>/dev/null | sed 's/X//' | head -1)"
        fi
        if [ -n "$ls_display" ] && [ -S "/tmp/.X11-unix/X${ls_display#:}" ]; then
          xauth_file="${XAUTHORITY:-$HOME/.Xauthority}"
          docker_base+=(-e DISPLAY="$ls_display"
            -v /tmp/.X11-unix:/tmp/.X11-unix
            -e XAUTHORITY="${XAUTHORITY:-/tmp/xauth}")
          if [ -f "$xauth_file" ]; then
            docker_base+=(-v "$xauth_file":/tmp/xauth)
          fi
          echo "[run_sim_grasp] native livestream: DISPLAY=$ls_display (grant with: xhost +local:)" >&2
        else
          echo "[run_sim_grasp] WARNING: no X display found for native livestream (mode 1)." >&2
          echo "[run_sim_grasp]   Start one: Xvfb :5 -screen 0 1920x1080x24 &  then LIVESTREAM_DISPLAY=:5" >&2
        fi
      fi
      docker_base+=(-v "$PWD":/work "${CACHE_MOUNTS[@]}" --entrypoint /isaac-sim/python.sh "$IMG")
      demo_timeout="${TIMEOUT:-2400}"
      # driver_args already contains whatever the user passed on the CLI
      # (including any --livestream / --livestream-hold). If they used the
      # LIVESTREAM_ON=1 shorthand instead of CLI flags, inject the flags here.
      # Always ensure --livestream-hold is present (defaults to LIVESTREAM_HOLD)
      # so grasp_demo holds long enough to connect a remote client.
      if [ "${LIVESTREAM_ON:-0}" = "1" ]; then
        driver_args+=(--livestream "$livestream_mode")
      fi
      has_ls_hold=0
      for a in "${driver_args[@]}"; do [ "$a" = "--livestream-hold" ] && has_ls_hold=1; done
      if [ "$has_ls_hold" = "0" ]; then
        driver_args+=(--livestream-hold "$livestream_hold")
      fi
    fi
    rm -f sim_grasp_demo_progress.txt
    timeout "$demo_timeout" "${docker_base[@]}" /work/src/novbts/sim/grasp_demo.py --mode "$mode" "${driver_args[@]}"
    chown_back
    if [ "$mode" = "pick" ]; then
      gate_marker GRASP_DEMO_OK sim_grasp_demo_progress.txt || exit 1
    else
      gate_marker PRESS_OK sim_grasp_demo_progress.txt || exit 1
    fi
    if [ -n "$tag" ]; then
      latest="runs/sim_grasp/$tag"
    else
      latest="$(ls -td runs/sim_grasp/${mode}_* 2>/dev/null | head -1)"
    fi
    test -n "$latest" || exit 1
    gate_npz "$latest/sequence.npz" || exit 1
    ;;
  *)
    echo "usage: bash infra/run_sim_grasp.sh check|build-asset|verify-asset|demo [press|pick] [args...]" >&2
    exit 2
    ;;
esac
