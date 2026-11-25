#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
case "${1:-main}" in
  pilot) datasets=(pickapic); extra=(--limit 1) ;;
  main) datasets=(pickapic geneval); extra=() ;;
  *) echo "Usage: bash scripts/run_768.sh [pilot|main]" >&2; exit 2 ;;
esac
for dataset in "${datasets[@]}"; do
  size=100
  if [[ "$dataset" == geneval ]]; then size=50; fi
  for method in satisdive sgi; do
    ".venvs/$method/bin/python" scripts/generate.py \
      --config "configs/single_gpu_768/$method.yaml" --dataset "$dataset" \
      --subset-size "$size" --device cuda:0 --reward-device 0 --resume "${extra[@]}"
  done
done
