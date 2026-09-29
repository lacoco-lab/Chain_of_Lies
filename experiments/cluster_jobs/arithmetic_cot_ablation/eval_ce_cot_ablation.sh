#!/bin/bash
# Evaluate one CE-only CoT-ablation condition for one seed.

set -e

CONDOR_USER="${1:-$USER}"
MODE="${2:-public_cot}"
VARIANT_NAME="${3:-arith_piggyback}"
SEED="${4:-0}"
ARTIFACTS_ROOT="${5:-artifacts/ce_cot_ablation}"
source /nethome/$CONDOR_USER/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "mode=$MODE variant=$VARIANT_NAME seed=$SEED"
echo "artifacts_root=$ARTIFACTS_ROOT"
nvidia-smi || true

INFERENCE_BATCH_SIZE=4
MAX_NEW_TOKENS=512
if [[ "$MODE" == "verbose_public_cot" ]]; then
  INFERENCE_BATCH_SIZE=1
  MAX_NEW_TOKENS=1024
fi
echo "inference_batch_size=$INFERENCE_BATCH_SIZE max_new_tokens=$MAX_NEW_TOKENS"

ARTIFACT_ROOT="$ARTIFACTS_ROOT/seed_$SEED/$MODE"
RESPONSES_ROOT="generated_data/eval_responses/$(basename "$ARTIFACTS_ROOT")/seed_$SEED/$MODE"
mkdir -p "$ARTIFACT_ROOT/per_variant_eval/ckpt_task"

python scripts/shared/evaluation/evaluate_ce.py \
  --artifacts-root "$ARTIFACT_ROOT" \
  --variant "$VARIANT_NAME" \
  --eval-responses-root "$RESPONSES_ROOT" \
  --model-responses-subdir trained \
  --checkpoint ckpt_task \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --greedy \
  --inference-batch-size "$INFERENCE_BATCH_SIZE" \
  --no-resume \
  --json-out "$ARTIFACT_ROOT/per_variant_eval/ckpt_task/$VARIANT_NAME.json"
