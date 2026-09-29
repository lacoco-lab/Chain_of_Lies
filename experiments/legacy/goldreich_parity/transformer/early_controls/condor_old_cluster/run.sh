#!/bin/bash

set -euo pipefail

CONDITION="${1:?Usage: run.sh CONDITION LENGTH DATA_ROOT}"
LENGTH="${2:?Usage: run.sh CONDITION LENGTH DATA_ROOT}"
DATA_ROOT="${3:?Usage: run.sh CONDITION LENGTH DATA_ROOT [CONFIG]}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/legacy/goldreich_parity/transformer/early_controls"
CONFIG="${4:-$EXPERIMENT/config.json}"
OUTPUT_ROOT="artifacts/parity_cot_controls/$CONDITION/n$LENGTH"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 "$EXPERIMENT/test_pipeline.py"
python3 "$EXPERIMENT/train.py" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --length "$LENGTH" \
  --condition "$CONDITION"

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
