#!/bin/bash
# Train one CE-only S5 condition for one seed.

set -e

CONDOR_USER="${1:-$USER}"
VARIANT_NAME="${2:-s5_piggyback}"
SUPERVISION_MODE="${3:-public_cot}"
SEED="${4:-0}"
EPOCHS="${5:-1}"
SAVE_EVERY="${6:-5000}"
EVAL_EVERY="${7:-5000}"
ARTIFACTS_ROOT="${8:-artifacts/ce_s5}"
source /nethome/$CONDOR_USER/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "variant=$VARIANT_NAME mode=$SUPERVISION_MODE seed=$SEED"
echo "artifacts_root=$ARTIFACTS_ROOT epochs=$EPOCHS save_every=$SAVE_EVERY eval_every=$EVAL_EVERY"
nvidia-smi || true

BATCH_SIZE=2
VALIDATION_BATCH_SIZE=4
VALIDATION_SAMPLE_SIZE=500
MAX_NEW_TOKENS=512
echo "train_batch_size=$BATCH_SIZE validation_batch_size=$VALIDATION_BATCH_SIZE validation_sample_size=$VALIDATION_SAMPLE_SIZE max_new_tokens=$MAX_NEW_TOKENS"

SPLIT_ROOT="generated_data/prompt_splits/seed_$SEED"
OUTPUT_ROOT="$ARTIFACTS_ROOT/seed_$SEED/$SUPERVISION_MODE"

if [[ ! -d "$SPLIT_ROOT/$VARIANT_NAME/train_prompts" ]]; then
  echo "Missing $SPLIT_ROOT/$VARIANT_NAME/train_prompts. Run experiments/cluster_jobs/s5_state_tracking/old_cluster/gen_splits_s5.sub first." >&2
  exit 1
fi

python scripts/shared/training/run_ce.py \
  --variant "$VARIANT_NAME" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-root "$OUTPUT_ROOT" \
  --split-root "$SPLIT_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate 2e-5 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --validation-sample-size "$VALIDATION_SAMPLE_SIZE" \
  --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --expected-train-prompts 10000 \
  --expected-val-prompts 1000 \
  --supervision-mode "$SUPERVISION_MODE" \
  --seed "$SEED"

