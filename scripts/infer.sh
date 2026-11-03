#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  set +u
  source /usr/local/PPU_SDK/envsetup.sh cuda >/dev/null
  set -u
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m wovrplus.inference "$@"
