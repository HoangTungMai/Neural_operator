#!/usr/bin/env bash
# Convergence-check FEM shear generation at hex-res 40 (vs the res-32 / res-24 sweeps).
# Purpose: settle whether res-32 is a CONVERGED tangential ground truth. The paired
# res-24 vs res-32 delta is large (tangential rel-L2 ~0.89), so res-32 itself may not
# be converged. res-40 (just under PhysX's ~res-48 stability ceiling) lets us measure
# the res-32 -> res-40 delta: small => res-32 trustworthy; still large => the tangential
# GT cannot be converged within PhysX's stable mesh range (a publishable finding).
#
# Combo list is IDENTICAL to gen_fem_sweep.sh / gen_fem_sweep_res32.sh (same pilots +
# same fixed seed 2024) so res-40 is PAIRED (same R, mu, E, seed per combo) with res-32.
# Separate output dir (data/fem/sweep40) so it never clobbers other sweeps. RESUMABLE.
#
# Usage: bash infra/gen_fem_sweep_res40.sh <NCOMBOS>   (each combo = 40 frames)
set -u
cd "$(dirname "$0")/.."
NCOMBOS="${1:-10}"                         # 10 pilot combos * 40 = 400 frames (paired subset)
IMG=isaac-lab-fem:latest
SCRIPT=/work/src/novbts/groundtruth/isaac_extract_shear.py
NAME=femsweep40
PY=.venv-gate2/bin/python
COMMON="--frames 40 --hex-res 40 --gel-xy 0.05 --gel-z 0.02 --marker-side 32"
mkdir -p data/fem/sweep40

# Reproducible combo list: IDENTICAL to gen_fem_sweep.sh (first 10 == pilot,
# rest sampled uniformly with the same fixed seed) so res-32 and res-40 share
# the same (R, mu, E, seed) per combo -> paired comparison.
$PY - "$NCOMBOS" > /tmp/femsweep40_combos.txt <<'PYEOF'
import sys, numpy as np
n = int(sys.argv[1])
pilot = [(0.020,0.6,1.0e5),(0.015,0.6,1.0e5),(0.025,0.6,1.0e5),(0.020,0.4,1.0e5),
         (0.020,0.8,1.0e5),(0.020,0.6,0.5e5),(0.020,0.6,2.0e5),(0.016,0.5,0.7e5),
         (0.024,0.8,1.6e5),(0.018,0.7,1.3e5)]
rng = np.random.default_rng(2024)
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
  out="/work/data/fem/sweep40/combo_$(printf '%03d' $idx)"
  host="data/fem/sweep40/combo_$(printf '%03d' $idx)/fem_gt_shear.npz"
  if [ -f "$host" ] && [ "$($PY -c "import numpy as np;print(np.load('$host',allow_pickle=True)['params'].shape[0])" 2>/dev/null)" = "40" ]; then
    echo "skip combo $idx (already 40 frames)"; ndone=$((ndone+1)); continue
  fi
  echo "=== combo $idx: R=$R mu=$MU E=$E seed=$SEED ==="
  docker rm -f $NAME >/dev/null 2>&1
  timeout 2400 docker run --rm --name $NAME --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh $IMG $SCRIPT $COMMON \
    --indentor-r $R --mu $MU --youngs $E --seed $SEED --out "$out"
  docker rm -f $NAME >/dev/null 2>&1
  # Container exit code is unreliable, so gate success on the npz holding 40 frames.
  nf="$($PY -c "import numpy as np;print(np.load('$host',allow_pickle=True)['params'].shape[0])" 2>/dev/null)"
  if [ "$nf" = "40" ]; then echo "combo $idx OK (40 frames)"; nrun=$((nrun+1));
  else echo "combo $idx FAILED (got '${nf:-0}' frames)"; nfail=$((nfail+1)); fi
done < /tmp/femsweep40_combos.txt
echo "SWEEP40 DONE: skipped=$ndone ran=$nrun failed=$nfail target=$NCOMBOS combos (x40 frames)"
