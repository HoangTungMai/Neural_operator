#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "FOUNDATION_ROUND2_START $(date -Is)"

for g in cylinder sphere_oodR cuboid ellipsoid bolt_hex rounded_tip cone multi_lobe; do
  echo "FOUNDATION_ROUND2_START_SHAPE $g $(date -Is)"
  IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh "$g" 24 25 3 12 23
  echo "FOUNDATION_ROUND2_DONE_SHAPE $g $(date -Is)"
done

echo "FOUNDATION_ROUND2_DONE $(date -Is)"
