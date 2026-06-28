#!/usr/bin/env bash
# UIPC GT TOP-UP sweep — appends combos 050+ to data/uipc/sweep/ to fix the two
# learning-curve findings on the 1998-frame clean IPC dataset:
#   (1) field rel-L2 saturates by ~1200 frames (2000 is enough for RQ1), BUT
#   (2) slip macro-F1 (head_a/head_b) is STILL climbing, and head_b is bottlenecked
#       by the normal class being only ~3% (63/1998).
# So the top-up is TARGETED, not just "more data":
#   - NORMAL combos (g=0, pure indentation) over an R x E grid  -> raises normal class
#   - EXTRA RANDOM combos (same sampling as gen_uipc_sweep.sh)  -> more stick/slip examples
# K=3 to match combos 005-049 (the swept dataset). One Isaac boot per combo (batch).
#
# Usage:  bash infra/gen_uipc_topup.sh [KREPS] [START_COMBO] [END_COMBO]
#   default KREPS=3, runs all top-up combos (050..062).
set -u
cd "$(dirname "$0")/.."

KREPS="${1:-3}"
START_COMBO="${2:-50}"
END_COMBO="${3:-62}"
IMG="${IMG:-isaac-lab-tacex:latest}"
SCRIPT=/work/src/novbts/groundtruth/tacex_uipc_extract_shear.py
NAME_PREFIX=uipctopup
PY=.venv-gate2/bin/python
SWEEP_DIR=data/uipc/sweep
ROWS_DIR="$SWEEP_DIR/_rows"
PROG="/work/fem_progress_uipc_topup.txt"
FRAMES=40

COMMON="--batch --gel-res 24 --eps-velocity 0.001 --gel-xy 0.10 --gel-z 0.04 \
  --marker-side 32 --press-steps 40 --settle-steps 10 --shear-steps 80 \
  --shear-settle 10 --batch-reps $KREPS --progress-file $PROG"

mkdir -p "$SWEEP_DIR" "$ROWS_DIR"

# Build the top-up combo meta (CI R MU E SEED KIND) AND per-combo rows files.
#   NORMAL: g=0 (no shear), depth~U(4,7)mm, R x E grid, mu fixed (irrelevant w/o shear).
#   RANDOM: depth~U(4,7)mm, g~U(0,1.3), random direction, |shear|=g*mu*0.01 (mirror sweep).
COMBO_META="$ROWS_DIR/_meta_topup.txt"
$PY - "$FRAMES" "$ROWS_DIR" > "$COMBO_META" <<'PYEOF'
import sys, numpy as np, os
frames, rows_dir = int(sys.argv[1]), sys.argv[2]
ci = 50
meta = []
# --- NORMAL combos: g=0 over R x E grid (9 combos: 050..058) ---
for R in (0.015, 0.020, 0.025):
    for E in (0.5e5, 1.0e5, 2.0e5):
        seed = 100 + ci
        rng = np.random.default_rng(seed)
        rows = []
        for fi in range(frames):
            depth = float(rng.uniform(0.004, 0.007))
            rows.append(f"{fi} {depth:.10g} 0 0 0")   # g=0 -> shear 0 -> mode 0 (normal)
        with open(os.path.join(rows_dir, f"combo_{ci:03d}.rows"), "w") as f:
            f.write("\n".join(rows) + "\n")
        meta.append((ci, R, 0.6, E, seed, "normal"))
        ci += 1
# --- EXTRA RANDOM combos: full g range, new R/mu/E (4 combos: 059..062) ---
box = np.random.default_rng(777)
for _ in range(4):
    R  = round(float(box.uniform(0.015, 0.025)), 4)
    mu = round(float(box.uniform(0.40, 0.80)), 3)
    E  = round(float(box.uniform(0.5e5, 2.0e5)), 0)
    seed = 100 + ci
    rng = np.random.default_rng(seed)
    rows = []
    for fi in range(frames):
        depth = float(rng.uniform(0.004, 0.007))
        g     = float(rng.uniform(0.0, 1.3))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        mag   = g * mu * 0.01
        rows.append(f"{fi} {depth:.10g} {g:.10g} {mag*np.cos(theta):.10g} {mag*np.sin(theta):.10g}")
    with open(os.path.join(rows_dir, f"combo_{ci:03d}.rows"), "w") as f:
        f.write("\n".join(rows) + "\n")
    meta.append((ci, R, mu, E, seed, "random"))
    ci += 1
for (c, R, mu, E, seed, kind) in meta:
    print(c, f"{R:.8g}", f"{mu:.8g}", f"{E:.8g}", seed, kind)
PYEOF

echo "UIPC TOP-UP: combos ${START_COMBO}..${END_COMBO} | frames=$FRAMES K=$KREPS | log $PROG"
ok=0; failc=0

while read -r CI R MU E SEED KIND; do
  [ "$CI" -lt "$START_COMBO" ] && continue
  [ "$CI" -gt "$END_COMBO" ] && continue
  combo="combo_$(printf '%03d' "$CI")"
  cdir="$SWEEP_DIR/$combo"
  rows="$ROWS_DIR/$combo.rows"
  mkdir -p "$cdir"
  echo "=== $combo [$KIND]: R=$R mu=$MU E=$E (batch ${FRAMES}x${KREPS} in ONE boot) ==="

  cname="${NAME_PREFIX}_${CI}"
  docker rm -f "$cname" >/dev/null 2>&1
  timeout 7200 docker run --rm --name "$cname" --gpus all \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 -v "$PWD":/work \
    --entrypoint /isaac-sim/python.sh "$IMG" "$SCRIPT" $COMMON \
    --indentor-r="$R" --mu="$MU" --youngs="$E" --seed="$SEED" \
    --batch-rows="/work/$rows" --out="/work/$cdir"
  docker rm -f "$cname" >/dev/null 2>&1

  # container writes as root -> chown back so host-side aggregate/reads work
  docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
    -c "chown -R $(id -u):$(id -g) /work/$cdir" >/dev/null 2>&1

  # aggregate each frame's K reps -> avg npz (gate on files, exit codes unreliable)
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

echo "TOP-UP DONE: combos_ok=$ok incomplete=$failc"
