#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# NUM_PROCESSES selects total cards; --set overrides a YAML parameter.
exec bash "$ROOT/scripts/train.sh" --config "$ROOT/configs/train/adjust_bottle_raymap_vace_100k.yaml" "$@"
