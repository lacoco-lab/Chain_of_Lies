#!/bin/bash

set -euo pipefail

LENGTH="${1:?Usage: run_standard_v2.sh LENGTH MODEL_DIM DATA_ROOT OUTPUT_ROOT}"
MODEL_DIM="${2:?Missing model width}"
DATA_ROOT="${3:?Missing data root}"
OUTPUT_ROOT="${4:?Missing output root}"
CONFIG="${5:-experiments/goldreich_parity/transformer/main_experiment/standard_v2_config.json}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python "$EXPERIMENT/test_pipeline.py"
python "$EXPERIMENT/train_standard_v2.py" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --length "$LENGTH" \
  --model-dim "$MODEL_DIM"

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
