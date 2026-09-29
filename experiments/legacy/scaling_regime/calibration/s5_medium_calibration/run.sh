#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/legacy/scaling_regime/calibration/s5_medium_calibration"
SPLIT_ROOT="generated_data/s5_medium_calibration"
RESPONSES_ROOT="generated_data/s5_medium_calibration_eval_responses"
ARTIFACTS_ROOT="artifacts/ce_s5_medium_calibration"
VARIANT="s5_control"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1

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

prepare_model_cache() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not available inside the job." >&2
    echo "Export it on the submit node before condor_submit; run_all.sub forwards only that variable." >&2
    exit 2
  fi
  echo "HF_TOKEN present: yes"
  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Authenticated model download/cache check: attempt $attempt/5"
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ['HF_TOKEN'])"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then
      echo "Model download failed; retrying after 30 seconds." >&2
      sleep 30
    fi
  done
  echo "Could not cache Qwen/Qwen2.5-7B-Instruct after five authenticated attempts." >&2
  exit 1
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  generate)
    mkdir -p "$SPLIT_ROOT/seed_0"
    python "$EXPERIMENT_DIR/generate_splits.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$SPLIT_ROOT"
    python "$EXPERIMENT_DIR/validate_splits.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --split-root "$SPLIT_ROOT"
    ;;

  train_eval)
    CONDITION="${1:?condition is required}"
    SUPERVISION_MODE="${2:?supervision mode is required}"
    FILLER_COUNT="${3:?filler count is required}"
    TRAIN_MAX_NEW_TOKENS="${4:?training max_new_tokens is required}"
    INFERENCE_MAX_NEW_TOKENS="${5:?inference max_new_tokens is required}"
    SEED=0
    ROOT="$ARTIFACTS_ROOT/seed_$SEED/$CONDITION"
    ADAPTER="$ROOT/$VARIANT"
    RESPONSES="$RESPONSES_ROOT/seed_$SEED/$CONDITION/$VARIANT"
    mkdir -p "$ADAPTER/per_variant_eval/ckpt_final" "$RESPONSES"
    bootstrap_python
    prepare_model_cache
    nvidia-smi

    FILLER_ARGS=()
    if [[ "$SUPERVISION_MODE" == "filler_public_cot" ]]; then
      FILLER_ARGS=(--filler-token-count "$FILLER_COUNT")
    elif [[ "$FILLER_COUNT" != "0" ]]; then
      echo "Non-filler condition received filler count $FILLER_COUNT" >&2
      exit 2
    fi

    python scripts/shared/training/run_ce.py \
      --variant "$VARIANT" \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output-root "$ROOT" \
      --split-root "$SPLIT_ROOT/seed_$SEED" \
      --epochs 3 \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --max-new-tokens "$TRAIN_MAX_NEW_TOKENS" \
      --save-every 10000 \
      --eval-every 10000 \
      --validation-sample-size 400 \
      --validation-batch-size 2 \
      --expected-train-prompts 10000 \
      --expected-val-prompts 2000 \
      --supervision-mode "$SUPERVISION_MODE" \
      "${FILLER_ARGS[@]}" \
      --seed "$SEED"

    python scripts/shared/evaluation/evaluate_ce.py \
      --artifacts-root "$ROOT" \
      --variant "$VARIANT" \
      --prompts-dir "$SPLIT_ROOT/seed_$SEED/$VARIANT/val_prompts" \
      --eval-responses-root "$RESPONSES" \
      --model-responses-subdir finetuned \
      --checkpoint ckpt_final \
      --max-new-tokens "$INFERENCE_MAX_NEW_TOKENS" \
      --temperature 0.0 \
      --greedy \
      --inference-batch-size 2 \
      --no-resume \
      --json-out "$ADAPTER/per_variant_eval/ckpt_final/$VARIANT.json"
    ;;

  summarize)
    mkdir -p "$ARTIFACTS_ROOT"
    touch \
      "$ARTIFACTS_ROOT/metrics_by_length.csv" \
      "$ARTIFACTS_ROOT/summary.json" \
      "$ARTIFACTS_ROOT/REPORT.md"
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --split-root "$SPLIT_ROOT" \
      --responses-root "$RESPONSES_ROOT" \
      --output-root "$ARTIFACTS_ROOT"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
