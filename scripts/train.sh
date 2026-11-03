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
export WOVR_RUN_TIMESTAMP=${WOVR_RUN_TIMESTAMP:-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}
export TOKENIZERS_PARALLELISM=false
export DS_ENV_FILE=/dev/null
PYTHON=${PYTHON:-python}  # use the activated environment, or an explicit interpreter
NUM_PROCESSES=${NUM_PROCESSES:-1}          # total cards across all nodes
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29502}
for arg in "$@"; do
  if [[ "$arg" == --dry-run || "$arg" == --validate-only || "$arg" == --help ]]; then
    exec "$PYTHON" -m wovrplus.training.cli "$@"
  fi
done
if [[ "$NUM_PROCESSES" == 1 ]]; then
  exec "$PYTHON" -m wovrplus.training.cli "$@"
fi
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$ROOT/configs/distributed/zero2.yaml}
"$PYTHON" -m wovrplus.training.distributed "$ACCELERATE_CONFIG"
exec "$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  -m wovrplus.training.cli "$@"
