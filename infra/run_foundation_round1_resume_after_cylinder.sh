#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "FOUNDATION_ROUND1_RESUME_AFTER_CYLINDER_START $(date -Is)"

for g in sphere_oodR cuboid ellipsoid bolt_hex rounded_tip; do
  echo "FOUNDATION_ROUND1_TOPUP_START $g $(date -Is)"
  IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh "$g" 12 25 3 6 11
  echo "FOUNDATION_ROUND1_TOPUP_DONE $g $(date -Is)"
done

for g in cone multi_lobe rounded_die gear pebble; do
  echo "FOUNDATION_ROUND1_NEW_START $g $(date -Is)"
  IMG="${IMG:-isaac-lab-tacex:latest}" bash infra/gen_uipc_geom_ood.sh "$g" 12 25 3 0 11
  echo "FOUNDATION_ROUND1_NEW_DONE $g $(date -Is)"
done

echo "FOUNDATION_ROUND1_RESUME_AFTER_CYLINDER_DONE $(date -Is)"
