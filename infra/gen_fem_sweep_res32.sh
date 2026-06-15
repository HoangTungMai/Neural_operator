#!/usr/bin/env bash
# Scale-up FEM shear generation at FINER mesh (hex-res 32 vs the res-24 sweep).
# Approach-3 lever to lower the tangential ceiling (0.146 relL2 / 14.8 deg on
# res-24, diagnosed as GT-fidelity bound). Same param box / combo list / seeds
# as gen_fem_sweep.sh so res-32 is a paired, drop-in finer-GT replacement.
#
# Separate output dir (data/fem/sweep32) so it never clobbers the res-24 sweep.
# RESUMABLE: a combo whose npz already holds 40 frames is skipped.
#
# Usage: bash infra/gen_fem_sweep_res32.sh <NCOMBOS>   (each combo = 40 frames)
set -u
cd "$(dirname "$0")/.."
NCOMBOS="${1:-50}"                         # 50 combos * 40 = 2000 frames
IMG=isaac-lab-fem:latest
SCRIPT=/work/src/novbts/groundtruth/isaac_extract_shear.py
NAME=femsweep32
PY=.venv-gate2/bin/python
COMMON="--frames 40 --hex-res 32 --gel-xy 0.05 --gel-z 0.02 --marker-side 32"
mkdir -p data/fem/sweep32

# Reproducible combo list: IDENTICAL to gen_fem_sweep.sh (first 10 == pilot,
# rest sampled uniformly with the same fixed seed) so res-24 and res-32 share
# the same (R, mu, E, seed) per combo -> paired comparison.
$PY - "$NCOMBOS" > /tmp/femsweep32_combos.txt <<'PYEOF'
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
  out="/work/data/fem/sweep32/combo_$(printf '%03d' $idx)"
  host="data/fem/sweep32/combo_$(printf '%03d' $idx)/fem_gt_shear.npz"
  if [ -f "$host" ] && [ "$($PY -c "import numpy as np;print(np.load('$host',allow_pickle=True)['params'].shape[0])" 2>/dev/null)" = "40" ]; then
    echo "skip combo $idx (already 40 frames)"; ndone=$((ndone+1)); continue
  fi
  echo "=== combo $idx: R=$R mu=$MU E=$E seed=$SEED ==="
  docker rm -f $NAME >/dev/null 2>&1
  timeout 1800 docker run --rm --name $NAME --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh $IMG $SCRIPT $COMMON \
    --indentor-r $R --mu $MU --youngs $E --seed $SEED --out "$out"
  docker rm -f $NAME >/dev/null 2>&1
  # Container exit code is unreliable (Isaac returns 0 even when an on-demand
  # dep install fails and no frames are written), so gate success on the npz
  # actually holding 40 frames.
  nf="$($PY -c "import numpy as np;print(np.load('$host',allow_pickle=True)['params'].shape[0])" 2>/dev/null)"
  if [ "$nf" = "40" ]; then echo "combo $idx OK (40 frames)"; nrun=$((nrun+1));
  else echo "combo $idx FAILED (got '${nf:-0}' frames)"; nfail=$((nfail+1)); fi
done < /tmp/femsweep32_combos.txt
echo "SWEEP32 DONE: skipped=$ndone ran=$nrun failed=$nfail target=$NCOMBOS combos (x40 frames)"
