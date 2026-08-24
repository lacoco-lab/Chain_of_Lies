#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/parity_prop4"
SPLIT_ROOT="generated_data/parity_prop4"
RESPONSES_ROOT="generated_data/parity_prop4_eval_responses"
ARTIFACTS_ROOT="artifacts/ce_parity_prop4"

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
  echo "Could not cache Qwen/Qwen2.5-7B-Instruct after five attempts." >&2
  exit 1
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  generate)
    python "$EXPERIMENT_DIR/generate_splits.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$SPLIT_ROOT"
    python "$EXPERIMENT_DIR/validate_splits.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --split-root "$SPLIT_ROOT"
    ;;

  train_eval)
    CONDITION="${1:?condition is required}"
    SEED=0
    ROOT="$ARTIFACTS_ROOT/seed_$SEED/$CONDITION"
    ADAPTER_ROOT="$ROOT/$CONDITION"
    PROMPTS="$SPLIT_ROOT/seed_$SEED/$CONDITION"
    RESPONSES="$RESPONSES_ROOT/seed_$SEED/$CONDITION"
    mkdir -p "$ROOT" "$RESPONSES/baseline" "$RESPONSES/finetuned"
    bootstrap_python
    prepare_model_cache
    nvidia-smi

    python scripts/run_ce_only.py \
      --variant "$CONDITION" \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output-root "$ROOT" \
      --train-prompts-dir "$PROMPTS/train_prompts" \
      --epochs 3 \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --max-new-tokens 256 \
      --save-every 10000 \
      --eval-every 10000 \
      --expected-train-prompts 10000 \
      --supervision-mode record_target \
      --seed "$SEED"

    cp "$ADAPTER_ROOT/training_metadata.json" "$ROOT/training_metadata.json"

    python "$EXPERIMENT_DIR/run_inference.py" \
      --prompts-dir "$PROMPTS/val_prompts" \
      --responses-dir "$RESPONSES/baseline" \
      --model Qwen/Qwen2.5-7B-Instruct \
      --max-new-tokens 256 \
      --batch-size 4

    python "$EXPERIMENT_DIR/run_inference.py" \
      --prompts-dir "$PROMPTS/val_prompts" \
      --responses-dir "$RESPONSES/finetuned" \
      --model "$ADAPTER_ROOT/ckpt_final" \
      --max-new-tokens 256 \
      --batch-size 4
    ;;

  summarize)
    mkdir -p "$ARTIFACTS_ROOT"
    touch "$ARTIFACTS_ROOT/metrics.csv" "$ARTIFACTS_ROOT/summary.json" "$ARTIFACTS_ROOT/REPORT.md"
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --split-root "$SPLIT_ROOT" \
      --responses-root "$RESPONSES_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
