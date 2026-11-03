#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# NUM_PROCESSES: total GPU/PPU count (default 1). ACCELERATE_CONFIG: zero0/1/2.yaml.
# Override individual YAML values with repeated --set dotted.key=value.
exec bash "$ROOT/scripts/train.sh" --config "$ROOT/configs/train/adjust_bottle_vector_cached_100k.yaml" "$@"
