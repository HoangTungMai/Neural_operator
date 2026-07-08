#!/usr/bin/env bash
# Isaac Sim UR3e/Robotiq/VBTS simulation runner.
set -u
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
PY="${PY:-.venv-gate2/bin/python}"
SUB="${1:-}"
shift || true

DOCKER_BASE=(docker run --rm --gpus all
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0
  -v "$PWD":/work
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
    for arg in "$@"; do
      if [ "$prev" = "--tag" ]; then
        tag="$arg"
        break
      fi
      prev="$arg"
    done
    rm -f sim_grasp_demo_progress.txt
    timeout "${TIMEOUT:-1800}" "${DOCKER_BASE[@]}" /work/src/novbts/sim/grasp_demo.py --mode "$mode" "$@"
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
