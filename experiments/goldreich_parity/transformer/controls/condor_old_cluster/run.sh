#!/bin/bash

set -euo pipefail

CONDITION="${1:?Usage: run.sh CONDITION LENGTH DATA_ROOT OUTPUT_ROOT}"
LENGTH="${2:?Missing length}"
DATA_ROOT="${3:?Missing data root}"
OUTPUT_ROOT="${4:?Missing output root}"
CONFIG="${5:-experiments/goldreich_parity/transformer/controls/config.json}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/controls"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python "$EXPERIMENT/test_pipeline.py"
python "$EXPERIMENT/train.py" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --length "$LENGTH" \
  --condition "$CONDITION"

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
