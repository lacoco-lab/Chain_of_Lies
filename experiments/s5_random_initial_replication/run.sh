#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/s5_random_initial_replication"
SPLIT_ROOT="generated_data/s5_random_initial_replication"
RESPONSES_ROOT="generated_data/s5_random_initial_replication_eval_responses"
ARTIFACTS_ROOT="artifacts/ce_s5_random_initial_replication"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$JOB_ROOT/hf_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bootstrap_python() {
  if [[ ! -d "$VENV" ]]; then
    python3 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    python -m pip install --no-cache-dir \
      'matplotlib>=3.8' \
      'transformers==4.44.2' \
      'accelerate==0.33.0' \
      'peft==0.12.0'
    return
  fi
  source "$VENV/bin/activate"
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  generate)
    SEED="${1:?seed is required}"
    python "$EXPERIMENT_DIR/generate_splits.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --seed "$SEED" \
      --variant all \
      --output-root "$SPLIT_ROOT"
    python "$EXPERIMENT_DIR/validate_splits.py" \
      --split-root "$SPLIT_ROOT/seed_$SEED" --variant s5_piggyback
    python "$EXPERIMENT_DIR/validate_splits.py" \
      --split-root "$SPLIT_ROOT/seed_$SEED" --variant s5_control
    ;;

  train)
    VARIANT="${1:?variant is required}"
    MODE="${2:?mode is required}"
    SEED="${3:?seed is required}"
    EPOCHS="${4:?epochs are required}"
    MAX_NEW_TOKENS="${5:?max_new_tokens is required}"
    SAVE_EVERY="${6:?save_every is required}"
    EVAL_EVERY="${7:?eval_every is required}"
    OUTPUT_ROOT="$ARTIFACTS_ROOT/seed_$SEED/$MODE"
    mkdir -p "$OUTPUT_ROOT/$VARIANT"
    bootstrap_python
    nvidia-smi
    python scripts/run_ce_only.py \
      --variant "$VARIANT" \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output-root "$OUTPUT_ROOT" \
      --split-root "$SPLIT_ROOT/seed_$SEED" \
      --epochs "$EPOCHS" \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --save-every "$SAVE_EVERY" \
      --eval-every "$EVAL_EVERY" \
      --validation-sample-size 200 \
      --validation-batch-size 1 \
      --expected-train-prompts 10000 \
      --expected-val-prompts 1000 \
      --supervision-mode "$MODE" \
      --seed "$SEED"
    ;;

  eval)
    VARIANT="${1:?variant is required}"
    MODE="${2:?mode is required}"
    SEED="${3:?seed is required}"
    MAX_NEW_TOKENS="${4:?max_new_tokens is required}"
    ROOT="$ARTIFACTS_ROOT/seed_$SEED/$MODE"
    ADAPTER="$ROOT/$VARIANT"
    RESPONSES="$RESPONSES_ROOT/seed_$SEED/$MODE/$VARIANT"
    mkdir -p "$ADAPTER/per_variant_eval/ckpt_final" "$RESPONSES"
    bootstrap_python
    nvidia-smi
    python scripts/evaluate_ce_model.py \
      --artifacts-root "$ROOT" \
      --variant "$VARIANT" \
      --prompts-dir "$SPLIT_ROOT/seed_$SEED/$VARIANT/val_prompts" \
      --eval-responses-root "$RESPONSES" \
      --model-responses-subdir finetuned \
      --checkpoint ckpt_final \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature 0.0 \
      --greedy \
      --inference-batch-size 2 \
      --no-resume \
      --json-out "$ADAPTER/per_variant_eval/ckpt_final/$VARIANT.json"
    ;;

  train_eval)
    VARIANT="${1:?variant is required}"
    MODE="${2:?mode is required}"
    SEED="${3:?seed is required}"
    EPOCHS="${4:?epochs are required}"
    TRAIN_MAX_NEW_TOKENS="${5:?training max_new_tokens is required}"
    SAVE_EVERY="${6:?save_every is required}"
    EVAL_EVERY="${7:?eval_every is required}"
    INFERENCE_MAX_NEW_TOKENS="${8:?inference max_new_tokens is required}"

    # Precreate both declared Condor outputs so a genuine training/runtime
    # failure is reported as a process error rather than transfer hold 12/2.
    mkdir -p \
      "$ARTIFACTS_ROOT/seed_$SEED/$MODE/$VARIANT" \
      "$RESPONSES_ROOT/seed_$SEED/$MODE/$VARIANT"

    bash "$EXPERIMENT_DIR/run.sh" train \
      "$VARIANT" "$MODE" "$SEED" "$EPOCHS" "$TRAIN_MAX_NEW_TOKENS" \
      "$SAVE_EVERY" "$EVAL_EVERY"
    bash "$EXPERIMENT_DIR/run.sh" eval \
      "$VARIANT" "$MODE" "$SEED" "$INFERENCE_MAX_NEW_TOKENS"
    ;;

  summarize)
    bootstrap_python
    python scripts/summarize_ce_results.py \
      --root "$ARTIFACTS_ROOT" \
      --modes answer_only,public_cot,verbose_public_cot \
      --variants s5_piggyback,s5_control \
      --piggyback-variant s5_piggyback \
      --control-variant s5_control
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
