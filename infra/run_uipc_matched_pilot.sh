#!/usr/bin/env bash
# Matched production-provenance pilot for the GT regen decision.
#
# Exact production anchors:
#   bottom=fixed, indentor constraint=30000, gel constraint=0,
#   eps_velocity=2.5e-5, d_hat=1e-4, resistance=1e9, gel_res=24.
# Three non-normal rows are read directly from the merged production NPZ.
# Each (frame, nz, tol) uses one Isaac boot and K in-process replicates.
set -u
set -o pipefail
cd "$(dirname "$0")/.."

IMG="${IMG:-isaac-lab-tacex:latest}"
SCRIPT=/work/src/novbts/research/groundtruth/tacex_uipc_extract_shear.py
PY=.venv-gate2/bin/python
DATA="${DATA:-data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz}"
OUT_ROOT="${OUT_ROOT:-data/uipc/matched_pilot_strength30000}"
FRAMES_STR="${FRAMES_STR:-1362 1812 2508}"
KREPS="${KREPS:-3}"
TOL_ANCHOR="${TOL_ANCHOR:-0.0003}"
TOL_MESH="${TOL_MESH:-0.00001}"
TOL_CHECK="${TOL_CHECK:-0.000001}"
EPS_VELOCITY="${EPS_VELOCITY:-0.000025}"
GEL_RES="${GEL_RES:-24}"
GEL_XY="${GEL_XY:-0.020}"
GEL_Z="${GEL_Z:-0.003}"
D_HAT="${D_HAT:-0.0001}"
CONTACT_RESISTANCE="${CONTACT_RESISTANCE:-1.0e9}"
GEL_BOTTOM_BC="${GEL_BOTTOM_BC:-fixed}"
GEL_CONSTRAINT_STRENGTH="${GEL_CONSTRAINT_STRENGTH:-0}"
INDENTOR_CONSTRAINT_STRENGTH="${INDENTOR_CONSTRAINT_STRENGTH:-30000}"
ROWS_DIR="$OUT_ROOT/_rows"
META_TSV="$OUT_ROOT/frame_meta.tsv"
TIMING_TSV="$OUT_ROOT/container_wall_times.tsv"

read -r -a FRAMES <<< "$FRAMES_STR"
mkdir -p "$ROWS_DIR"

"$PY" - "$DATA" "$ROWS_DIR" "$META_TSV" "${FRAMES[@]}" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np

data_path, rows_dir, meta_path, *frame_args = sys.argv[1:]
frames = [int(x) for x in frame_args]
rows_dir = Path(rows_dir)
rows_dir.mkdir(parents=True, exist_ok=True)

with np.load(data_path, allow_pickle=True) as data:
    params = np.asarray(data["params"], dtype=np.float64)
    modes = np.asarray(data["mode"], dtype=np.int32)
    load_modes = np.asarray(data["load_mode"], dtype=np.int32) if "load_mode" in data.files else None
    load_names = str(np.asarray(data["load_mode_names"]).item()).split(",") \
        if "load_mode_names" in data.files else ["linear", "ortho", "reverse"]
    lines = ["frame\tmode\tR\tmu\tE\tdepth\tg\tsx\tsy\tload_mode"]
    for frame in frames:
        p = params[frame]
        sx, sy, mu = float(p[4]), float(p[5]), float(p[6])
        g = float(np.hypot(sx, sy) / (max(mu, 1e-12) * 1e-3))
        load_mode = load_names[int(load_modes[frame])] if load_modes is not None else "linear"
        row = rows_dir / f"f{frame}.rows"
        row.write_text(
            f"{frame} {p[2]:.17g} {g:.17g} {sx:.17g} {sy:.17g} {load_mode}\n"
        )
        lines.append(
            f"{frame}\t{int(modes[frame])}\t{p[3]:.17g}\t{mu:.17g}\t{p[7]:.17g}\t"
            f"{p[2]:.17g}\t{g:.17g}\t{sx:.17g}\t{sy:.17g}\t{load_mode}"
        )
Path(meta_path).write_text("\n".join(lines) + "\n")
PYEOF

SETTINGS=(
  "4 $TOL_ANCHOR anchor"
  "4 $TOL_MESH mesh"
  "8 $TOL_MESH mesh"
  "16 $TOL_MESH mesh"
  "24 $TOL_MESH mesh"
  "24 $TOL_CHECK tolcheck"
)

echo "MATCHED PILOT: frames=$FRAMES_STR K=$KREPS out=$OUT_ROOT"
echo "config: gel_res=$GEL_RES eps=$EPS_VELOCITY bottom=$GEL_BOTTOM_BC gel_strength=$GEL_CONSTRAINT_STRENGTH indentor_strength=$INDENTOR_CONSTRAINT_STRENGTH"

while IFS=$'\t' read -r FI MODE R MU E DEPTH G SX SY LOAD_MODE; do
  for spec in "${SETTINGS[@]}"; do
    read -r NZ VTOL PURPOSE <<< "$spec"
    setting="nz${NZ}_tol${VTOL}"
    out_host="$OUT_ROOT/f${FI}/$setting"
    out_container="/work/$out_host"
    rows_container="/work/$ROWS_DIR/f${FI}.rows"
    frame_dir="$out_host/frame_$(printf '%03d' "$FI")"
    avg="$frame_dir/uipc_gt_shear_avg.npz"
    n_before="$(find "$frame_dir" -path '*/rep_*/uipc_gt_shear.npz' -type f 2>/dev/null | wc -l)"

    if [ "$n_before" -ge "$KREPS" ] && [ -f "$avg" ]; then
      echo "skip f$FI $setting ($KREPS reps + avg exist)"
      continue
    fi

    tol_tag="${VTOL//./p}"; tol_tag="${tol_tag//-/m}"
    cname="uipcmp_f${FI}_nz${NZ}_${tol_tag}"
    prog="/work/fem_progress_uipc_matched_f${FI}_nz${NZ}_${tol_tag}.txt"
    echo "=== f$FI mode=$MODE nz=$NZ tol=$VTOL purpose=$PURPOSE reps=$n_before/$KREPS ==="
    docker rm -f "$cname" >/dev/null 2>&1
    wall_start_ns="$(date +%s%N)"
    timeout 7200 docker run --rm --name "$cname" --gpus all \
      -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e LIVESTREAM=0 -v "$PWD":/work \
      --entrypoint /isaac-sim/python.sh "$IMG" "$SCRIPT" \
      --batch --batch-rows "$rows_container" --batch-reps "$KREPS" \
      --out "$out_container" --progress-file "$prog" \
      --gel-res "$GEL_RES" --gel-nz "$NZ" --eps-velocity "$EPS_VELOCITY" \
      --gel-xy "$GEL_XY" --gel-z "$GEL_Z" --indentor-r "$R" --mu "$MU" --youngs "$E" \
      --d-hat "$D_HAT" --contact-resistance "$CONTACT_RESISTANCE" \
      --velocity-tol "$VTOL" --gel-bottom-bc "$GEL_BOTTOM_BC" \
      --gel-constraint-strength "$GEL_CONSTRAINT_STRENGTH" \
      --indentor-constraint-strength "$INDENTOR_CONSTRAINT_STRENGTH" \
      --marker-side 32 --press-steps 40 --settle-steps 10 \
      --shear-steps 80 --shear-settle 10 --seed "$FI"
    docker rm -f "$cname" >/dev/null 2>&1
    wall_end_ns="$(date +%s%N)"

    docker run --rm -v "$PWD":/work --entrypoint bash "$IMG" \
      -c "chown -R $(id -u):$(id -g) /work/$OUT_ROOT" >/dev/null 2>&1

    n_after="$(find "$frame_dir" -path '*/rep_*/uipc_gt_shear.npz' -type f 2>/dev/null | wc -l)"
    wall_s="$($PY -c "print(($wall_end_ns-$wall_start_ns)/1e9)")"
    "$PY" - "$TIMING_TSV" "$FI" "$NZ" "$VTOL" "$n_before" "$n_after" "$wall_s" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
header = "frame\tnz\tvelocity_tol\treps_before\treps_after\tcontainer_wall_s\n"
row = "\t".join(sys.argv[2:]) + "\n"
if not path.exists():
    path.write_text(header + row)
else:
    with path.open("a") as handle:
        handle.write(row)
PYEOF

    if [ "$n_after" -ne "$KREPS" ]; then
      echo "FAILED f$FI $setting: only $n_after/$KREPS valid replicate files"
      exit 1
    fi

    "$PY" - "$DATA" "$FI" "$NZ" "$VTOL" "$EPS_VELOCITY" \
      "$GEL_BOTTOM_BC" "$GEL_CONSTRAINT_STRENGTH" \
      "$INDENTOR_CONSTRAINT_STRENGTH" "$frame_dir" <<'PYEOF'
import glob
import sys

import numpy as np

data_path, frame, nz, vtol, eps, bc, gel_strength, ind_strength, frame_dir = sys.argv[1:]
frame, nz = int(frame), int(nz)
with np.load(data_path, allow_pickle=True) as production:
    expected_params = np.asarray(production["params"][frame, :9], dtype=np.float64)
for path in sorted(glob.glob(f"{frame_dir}/rep_*/uipc_gt_shear.npz")):
    with np.load(path, allow_pickle=True) as data:
        got_params = np.asarray(data["params"][0, :9], dtype=np.float64)
        assert np.allclose(got_params, expected_params, rtol=1e-6, atol=1e-10), path
        checks = {
            "gel_res": 24,
            "gel_nz": nz,
            "eps_velocity": float(eps),
            "velocity_tol": float(vtol),
            "gel_bottom_bc": bc,
            "gel_constraint_strength": float(gel_strength),
            "indentor_constraint_strength": float(ind_strength),
        }
        for key, want in checks.items():
            got = np.asarray(data[key]).reshape(-1)[0]
            if isinstance(want, str):
                assert str(got) == want, (path, key, got, want)
            else:
                assert np.isclose(float(got), want, rtol=1e-6, atol=1e-12), (path, key, got, want)
        assert "run_time_s" in data.files, f"{path}: missing run_time_s"
        assert np.isfinite(data["disp"]).all(), path
PYEOF

    "$PY" -m novbts.research.groundtruth.aggregate_uipc_replicates \
      --glob "$frame_dir/rep_*/uipc_gt_shear.npz" --out "$avg" \
      --mode-shear-scale 0.001
    echo "OK f$FI $setting wall=${wall_s}s -> $avg"
  done
done < <(tail -n +2 "$META_TSV")

"$PY" infra/analyze_uipc_matched_pilot.py \
  --data "$DATA" --root "$OUT_ROOT" --frames $FRAMES_STR \
  --tol-anchor "$TOL_ANCHOR" --tol-mesh "$TOL_MESH" --tol-check "$TOL_CHECK"
