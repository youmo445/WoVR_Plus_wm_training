#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  set +u; source /usr/local/PPU_SDK/envsetup.sh cuda >/dev/null; set -u
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PYTHON=${PYTHON:-python}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NUM_PROCESSES:-1}" \
  -m wovrplus.data.build_latent_cache "$@"
