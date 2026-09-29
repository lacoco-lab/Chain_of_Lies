#!/bin/bash

set -euo pipefail

LENGTH="${1:?Usage: run_scale.sh LENGTH}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"
CONFIG="$EXPERIMENT/scale_config.json"
DATA_ROOT="generated_data/parity_goldreich_toy_transformer_scale/seed_0"
OUTPUT_ROOT="artifacts/parity_goldreich_toy_transformer_scale/n$LENGTH"

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
  --supervision full

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
