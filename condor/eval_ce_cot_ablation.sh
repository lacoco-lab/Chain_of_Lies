#!/bin/bash
# Evaluate one CE-only CoT-ablation condition for one seed.

set -e

CONDOR_USER="${1:-$USER}"
MODE="${2:-public_cot}"
VARIANT_NAME="${3:-arith_piggyback}"
SEED="${4:-0}"
source /nethome/$CONDOR_USER/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Host: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "mode=$MODE variant=$VARIANT_NAME seed=$SEED"
nvidia-smi || true

ARTIFACT_ROOT="artifacts/ce_cot_ablation/seed_$SEED/$MODE"
RESPONSES_ROOT="data/ce_cot_ablation_eval_responses/seed_$SEED/$MODE"
mkdir -p "$ARTIFACT_ROOT/per_variant_eval/ckpt_task"

python scripts/evaluate_rule_based_rl.py   --artifacts-root "$ARTIFACT_ROOT"   --variant "$VARIANT_NAME"   --eval-responses-root "$RESPONSES_ROOT"   --model-responses-subdir trained   --checkpoint ckpt_task   --max-new-tokens 512   --temperature 0.0   --greedy   --inference-batch-size 4   --no-resume   --json-out "$ARTIFACT_ROOT/per_variant_eval/ckpt_task/$VARIANT_NAME.json"
