#!/usr/bin/env bash
# Production UIPC GT sweep for the FNO pipeline — BATCH-per-combo (one Isaac boot
# per combo, the driver loops all frames x K reps in-process to amortise the
# ~23s/run Isaac boot). Validated: batch fields match per-boot within solver noise,
# no state-bleed, no GPU leak.
#
# Layout:
#   data/uipc/sweep/combo_000/frame_000/rep_1/uipc_gt_shear.npz
#   data/uipc/sweep/combo_000/frame_000/uipc_gt_shear_avg.npz   (K reps averaged)
#
# K reps are averaged because libuipc GPU runs are NOT deterministic (tangential
# run-to-run noise ~2.4% at res24). K=3 -> mean noise ~1.4% (K=6 ~1.0%); the gap
# is negligible vs the ~5-6% mesh-convergence residual, so K=3 is the default.
#
# Usage:
#   bash infra/gen_uipc_sweep.sh <NCOMBOS> <FRAMES> <K_REPS> [START_COMBO] [END_COMBO]
# Two non-overlapping shards (run in parallel on one GPU, ~1.5x):
#   bash infra/gen_uipc_sweep.sh 50 40 3 5 26
#   bash infra/gen_uipc_sweep.sh 50 40 3 27 49
# (combos 0-4 already exist at K=6 from the earlier run — left untouched.)
set -u
cd "$(dirname "$0")/.."

NCOMBOS="${1:-50}"
FRAMES="${2:-40}"
KREPS="${3:-3}"
START_COMBO="${4:-0}"
END_COMBO="${5:-$((NCOMBOS - 1))}"
IMG="${IMG:-isaac-lab-tacex:latest}"
SCRIPT=/work/src/novbts/groundtruth/tacex_uipc_extract_shear.py
NAME_PREFIX=uipcsweep
PY=.venv-gate2/bin/python
SWEEP_DIR=data/uipc/sweep
ROWS_DIR="$SWEEP_DIR/_rows"
SHARD="${START_COMBO}_${END_COMBO}"
PROG="/work/fem_progress_uipc_${SHARD}.txt"   # per-shard log so parallel shards don't clobber

COMMON="--batch --gel-res 24 --eps-velocity 0.001 --gel-xy 0.10 --gel-z 0.04 \
        --marker-side 32 --press-steps 40 --settle-steps 10 --shear-steps 80 \
        --shear-settle 10 --batch-reps $KREPS --progress-file $PROG"

mkdir -p "$SWEEP_DIR" "$ROWS_DIR"

# Generate per-combo metadata (R mu E) + per-combo rows files (frame depth g sx sy).
# Sampling mirrors isaac_extract_shear.py: depth~U(4,7)mm, g~U(0,1.3),
# shear=g*mu*0.01 m, random direction. Combo list mirrors gen_fem_sweep.sh.
COMBO_META="$ROWS_DIR/_meta_${SHARD}.txt"
$PY - "$NCOMBOS" "$FRAMES" "$START_COMBO" "$END_COMBO" "$ROWS_DIR" > "$COMBO_META" <<'PYEOF'
import sys, numpy as np, os
n, frames, start, end, rows_dir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
pilot = [(0.020,0.6,1.0e5),(0.015,0.6,1.0e5),(0.025,0.6,1.0e5),(0.020,0.4,1.0e5),
         (0.020,0.8,1.0e5),(0.020,0.6,0.5e5),(0.020,0.6,2.0e5),(0.016,0.5,0.7e5),
         (0.024,0.8,1.6e5),(0.018,0.7,1.3e5)]
box = np.random.default_rng(2024)
for ci in range(n):
    if ci < len(pilot):
        R, mu, E = pilot[ci]
    else:
        R = round(float(box.uniform(0.015, 0.025)), 4)
        mu = round(float(box.uniform(0.40, 0.80)), 3)
        E = round(float(box.uniform(0.5e5, 2.0e5)), 0)
    if ci < start or ci > end:
        continue
    rng = np.random.default_rng(42 + ci)
    rows = []
    for fi in range(frames):
        depth = float(rng.uniform(0.004, 0.007))
        g = float(rng.uniform(0.0, 1.3))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = g * mu * 0.01
        rows.append(f"{fi} {depth:.10g} {g:.10g} {mag*np.cos(theta):.10g} {mag*np.sin(theta):.10g}")
    with open(os.path.join(rows_dir, f"combo_{ci:03d}.rows"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(ci, f"{R:.8g}", f"{mu:.8g}", f"{E:.8g}", 42 + ci)
PYEOF

echo "UIPC BATCH SWEEP: combos ${START_COMBO}..${END_COMBO} | frames=$FRAMES K=$KREPS | log $PROG"
ok=0; failc=0

while read -r CI R MU E SEED; do
  combo="combo_$(printf '%03d' "$CI")"
  cdir="$SWEEP_DIR/$combo"
  rows="$ROWS_DIR/$combo.rows"
  mkdir -p "$cdir"
  echo "=== $combo: R=$R mu=$MU E=$E (batch ${FRAMES}x${KREPS} in ONE boot) ==="

  # one Isaac boot for the whole combo (resumable: --batch skips existing rep npz)
  cname="${NAME_PREFIX}_${CI}"
  docker rm -f "$cname" >/dev/null 2>&1
  timeout 7200 docker run --rm --name "$cname" --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh "$IMG" "$SCRIPT" $COMMON \
    --indentor-r="$R" --mu="$MU" --youngs="$E" --seed="$SEED" \
    --batch-rows="/work/$rows" --out="/work/$cdir"
  docker rm -f "$cname" >/dev/null 2>&1

  # the container writes as root -> chown back to the host user so the host-side
  # aggregate (and later reads) can write/read the combo dir.
  docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(id -u):$(id -g) /work/$cdir" >/dev/null 2>&1

  # aggregate each frame's K reps -> avg npz (container exit code unreliable; gate on files)
  fdone=0
  for fi in $(seq 0 $((FRAMES - 1))); do
    fdir="$cdir/frame_$(printf '%03d' "$fi")"
    avg="$fdir/uipc_gt_shear_avg.npz"
    nrep="$($PY -c "import glob;print(len(glob.glob('$fdir/rep_*/uipc_gt_shear.npz')))" 2>/dev/null)"
    if [ "$nrep" = "$KREPS" ]; then
      [ -f "$avg" ] || $PY -m novbts.groundtruth.aggregate_uipc_replicates \
        --glob "$fdir/rep_*/uipc_gt_shear.npz" --out "$avg" --mode-shear-scale 0.01 >/dev/null 2>&1
      [ -f "$avg" ] && fdone=$((fdone+1))
    fi
  done
  if [ "$fdone" = "$FRAMES" ]; then echo "  $combo OK ($fdone/$FRAMES frames)"; ok=$((ok+1))
  else echo "  $combo INCOMPLETE ($fdone/$FRAMES frames)"; failc=$((failc+1)); fi
done < "$COMBO_META"

echo "BATCH SWEEP SHARD ${SHARD} DONE: combos_ok=$ok incomplete=$failc"
