#!/bin/bash

set -euo pipefail

MODEL_DIM="${1:?Usage: run_standard_v2_n2048.sh MODEL_DIM}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"
DATA_ROOT="generated_data/parity_goldreich_toy_transformer_scale/seed_0"
OUTPUT_ROOT="artifacts/parity_goldreich_standard_transformer_v2/d${MODEL_DIM}/n2048"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python "$EXPERIMENT/test_pipeline.py"
python "$EXPERIMENT/train_standard_v2.py" \
  --config "$EXPERIMENT/standard_v2_n2048_config.json" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --length 2048 \
  --model-dim "$MODEL_DIM" \
  --continue-after-stage-gate-failure \
  --allow-failed-final-gate

test -s "$OUTPUT_ROOT/metrics.json"
test -s "$OUTPUT_ROOT/model.pt"
echo "Completed final N=2048 evaluation for d_model=$MODEL_DIM"
