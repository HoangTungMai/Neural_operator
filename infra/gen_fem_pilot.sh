#!/usr/bin/env bash
# Generate a parameter-swept fine-mesh (res-24) FEM shear training set.
# One (R, mu, E) combo per container (isolates Isaac long-run flakiness);
# each combo runs 40 frames varying depth + shear. Geometry fixed at the
# user-confirmed 50x50x20mm gel, marker grid 32x32.
#
# Robustness: per-combo timeout + forced container cleanup so a single hang
# doesn't stall the rest (only 1 container fits in 15GB RAM).
set -u
cd "$(dirname "$0")/.."           # repo root
IMG=isaac-lab-fem:latest
SCRIPT=/work/src/novbts/groundtruth/isaac_extract_shear.py
NAME=fempilot
COMMON="--frames 40 --hex-res 24 --gel-xy 0.05 --gel-z 0.02 --marker-side 32"

# combo: R  MU  E   (10 combos ~ 400 frames; center + one-at-a-time + mixed corners)
COMBOS=(
  "0.020 0.6 1.0e5"
  "0.015 0.6 1.0e5"
  "0.025 0.6 1.0e5"
  "0.020 0.4 1.0e5"
  "0.020 0.8 1.0e5"
  "0.020 0.6 0.5e5"
  "0.020 0.6 2.0e5"
  "0.016 0.5 0.7e5"
  "0.024 0.8 1.6e5"
  "0.018 0.7 1.3e5"
)

i=0
for c in "${COMBOS[@]}"; do
  read R MU E <<< "$c"
  seed=$((42 + i))
  out="/work/data/fem/pilot/combo_$(printf '%02d' $i)"
  echo "=== combo $i: R=$R mu=$MU E=$E seed=$seed -> $out ==="
  docker rm -f $NAME >/dev/null 2>&1
  timeout 900 docker run --rm --name $NAME --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh $IMG $SCRIPT $COMMON \
    --indentor-r $R --mu $MU --youngs $E --seed $seed --out "$out" \
    && echo "combo $i OK" || echo "combo $i FAILED/timeout (rc=$?)"
  docker rm -f $NAME >/dev/null 2>&1
  i=$((i + 1))
done
echo "ALL COMBOS DONE"
