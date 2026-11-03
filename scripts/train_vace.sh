#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Compatibility entry; prefer train_raymap_vace_cached.sh.
exec bash "$ROOT/scripts/train_raymap_vace_cached.sh" "$@"
