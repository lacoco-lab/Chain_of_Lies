#!/bin/bash
# Train one CE-only CoT-ablation condition for one seed.

set -e

CONDOR_USER="${1:-$USER}"
VARIANT_NAME="${2:-arith_piggyback}"
SUPERVISION_MODE="${3:-public_cot}"
SEED="${4:-0}"
EPOCHS="${5:-1}"
SAVE_EVERY="${6:-1000}"
EVAL_EVERY="${7:-1000}"
source /nethome/$CONDOR_USER/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "variant=$VARIANT_NAME mode=$SUPERVISION_MODE seed=$SEED"
nvidia-smi || true

SPLIT_ROOT="data/RL_splits/seed_$SEED"
OUTPUT_ROOT="artifacts/ce_cot_ablation/seed_$SEED/$SUPERVISION_MODE"

if [[ ! -d "$SPLIT_ROOT/$VARIANT_NAME/train_prompts" ]]; then
  echo "Missing $SPLIT_ROOT/$VARIANT_NAME/train_prompts. Run condor/gen_splits_correlated_pair.sub first." >&2
  exit 1
fi

python scripts/run_ce_only.py   --variant "$VARIANT_NAME"   --model Qwen/Qwen2.5-7B-Instruct   --output-root "$OUTPUT_ROOT"   --split-root "$SPLIT_ROOT"   --epochs "$EPOCHS"   --batch-size 2   --learning-rate 2e-5   --max-new-tokens 256   --save-every "$SAVE_EVERY"   --eval-every "$EVAL_EVERY"   --validation-sample-size 1000   --validation-batch-size 4   --supervision-mode "$SUPERVISION_MODE"   --seed "$SEED"
