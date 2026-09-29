#!/bin/bash

set -euo pipefail

LENGTH="${1:?Usage: run_standard.sh LENGTH MODEL_DIM DATA_ROOT}"
MODEL_DIM="${2:?Missing model width}"
DATA_ROOT="${3:?Missing data root}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"
CONFIG="$EXPERIMENT/standard_config.json"
OUTPUT_ROOT="artifacts/parity_goldreich_standard_transformer/d${MODEL_DIM}/n${LENGTH}"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python "$EXPERIMENT/test_pipeline.py"
python "$EXPERIMENT/train_standard.py" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --length "$LENGTH" \
  --model-dim "$MODEL_DIM"

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
