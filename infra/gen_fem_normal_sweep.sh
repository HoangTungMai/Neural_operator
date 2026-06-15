#!/usr/bin/env bash
# Pure-NORMAL FEM frames (drive ratio g=0 -> mode 0) to balance the slip class
# and rescue the separate slip-classifier (head-b), whose normal-F1 collapses on
# the shear sweep (g~U(0,1.3) gives only ~3% normal frames).
#
# Same param-combo list/seeds as gen_fem_sweep.sh (paired R/mu/E box), same fine
# mesh (res-24) and marker grid (side=32) so frames merge with shear_fine_swept.
# Pure-normal needs no tangential drag -> lateral-steps cut to 2 (~6x faster/frame).
#
# RESUMABLE: a combo whose npz already holds 40 frames is skipped.
# Usage: bash infra/gen_fem_normal_sweep.sh <NCOMBOS>   (each combo = 40 frames)
set -u
cd "$(dirname "$0")/.."
NCOMBOS="${1:-10}"                         # 10 combos * 40 = 400 normal frames
IMG=isaac-lab-fem:latest
SCRIPT=/work/src/novbts/groundtruth/isaac_extract_shear.py
NAME=femnormsweep
PY=.venv-gate2/bin/python
COMMON="--frames 40 --hex-res 24 --gel-xy 0.05 --gel-z 0.02 --marker-side 32 \
        --gmin 0 --gmax 0 --lateral-steps 2 --lateral-settle 1"
mkdir -p data/fem/normal_sweep

# Reproducible combo list: identical to gen_fem_sweep.sh (pilot first, then box).
$PY - "$NCOMBOS" > /tmp/femnormsweep_combos.txt <<'PYEOF'
import sys, numpy as np
n = int(sys.argv[1])
pilot = [(0.020,0.6,1.0e5),(0.015,0.6,1.0e5),(0.025,0.6,1.0e5),(0.020,0.4,1.0e5),
         (0.020,0.8,1.0e5),(0.020,0.6,0.5e5),(0.020,0.6,2.0e5),(0.016,0.5,0.7e5),
         (0.024,0.8,1.6e5),(0.018,0.7,1.3e5)]
rng = np.random.default_rng(2024)
rows = []
for i in range(n):
    if i < len(pilot):
        R, mu, E = pilot[i]
    else:
        R  = round(float(rng.uniform(0.015, 0.025)), 4)
        mu = round(float(rng.uniform(0.40, 0.80)), 3)
        E  = round(float(rng.uniform(0.5e5, 2.0e5)), 0)
    print(f"{i} {R} {mu} {E:.6g} {42+i}")
PYEOF

ndone=0; nrun=0; nfail=0
while read idx R MU E SEED; do
  out="/work/data/fem/normal_sweep/combo_$(printf '%03d' $idx)"
  host="data/fem/normal_sweep/combo_$(printf '%03d' $idx)/fem_gt_shear.npz"
  if [ -f "$host" ] && [ "$($PY -c "import numpy as np;print(np.load('$host',allow_pickle=True)['params'].shape[0])" 2>/dev/null)" = "40" ]; then
    echo "skip combo $idx (already 40 frames)"; ndone=$((ndone+1)); continue
  fi
  echo "=== normal combo $idx: R=$R mu=$MU E=$E seed=$SEED ==="
  docker rm -f $NAME >/dev/null 2>&1
  timeout 900 docker run --rm --name $NAME --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh $IMG $SCRIPT $COMMON \
    --indentor-r $R --mu $MU --youngs $E --seed $SEED --out "$out" \
    && { echo "combo $idx OK"; nrun=$((nrun+1)); } || { echo "combo $idx FAILED (rc=$?)"; nfail=$((nfail+1)); }
  docker rm -f $NAME >/dev/null 2>&1
done < /tmp/femnormsweep_combos.txt
echo "NORMAL SWEEP DONE: skipped=$ndone ran=$nrun failed=$nfail target=$NCOMBOS combos (x40 frames)"
