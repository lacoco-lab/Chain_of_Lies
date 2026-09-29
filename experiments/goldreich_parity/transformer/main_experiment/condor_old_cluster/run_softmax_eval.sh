#!/bin/bash

set -euo pipefail

LENGTH="${1:?Usage: run_softmax_eval.sh LENGTH CONFIG DATA_ROOT CHECKPOINT_ROOT}"
CONFIG="${2:?Missing config path}"
DATA_ROOT="${3:?Missing data root}"
CHECKPOINT_ROOT="${4:?Missing checkpoint root}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"
OUTPUT="artifacts/parity_goldreich_toy_transformer_softmax/n${LENGTH}/metrics.json"

cd "$JOB_ROOT"
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONUNBUFFERED=1

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python "$EXPERIMENT/evaluate_softmax.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_ROOT/n${LENGTH}/model.pt" \
  --original-metrics "$CHECKPOINT_ROOT/n${LENGTH}/metrics.json" \
  --test-data "$DATA_ROOT/n${LENGTH}/test.jsonl.gz" \
  --output "$OUTPUT" \
  --length "$LENGTH"

test -s "$OUTPUT"
