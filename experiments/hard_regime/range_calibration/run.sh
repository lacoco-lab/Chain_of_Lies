#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/hard_regime/range_calibration"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  local model_key="${1:?model key is required}"
  local model_id
  local token_argument="None"
  if [[ "$model_key" == "llama" ]]; then
    model_id="meta-llama/Llama-3.1-8B-Instruct"
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "HF_TOKEN is required for Llama; accept its license and export the token before submission." >&2
      exit 2
    fi
    token_argument="os.environ['HF_TOKEN']"
  elif [[ "$model_key" == "qwen" ]]; then
    model_id="Qwen/Qwen2.5-7B-Instruct"
  else
    echo "Unknown model key: $model_key" >&2
    exit 2
  fi

  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Model cache check for $model_id: attempt $attempt/5"
    if MODEL_ID="$model_id" TOKEN_ARGUMENT="$token_argument" python -c \
      "import os; from huggingface_hub import snapshot_download; token = os.environ.get('HF_TOKEN') if os.environ['TOKEN_ARGUMENT'] != 'None' else None; snapshot_download(os.environ['MODEL_ID'], token=token)"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then
      echo "Model download failed; retrying after 20 seconds." >&2
      sleep 20
    fi
  done
  echo "Could not cache $model_id." >&2
  exit 1
}

case "$ACTION" in
  generate|validate)
    FAMILY="${1:?family is required}"
    RANGE_ID="${2:?range is required}"
    python3 "$EXPERIMENT_DIR/experiment.py" "$ACTION" --family "$FAMILY" --range "$RANGE_ID"
    ;;
  train|eval)
    MODEL_KEY="${1:?model is required}"
    FAMILY="${2:?family is required}"
    RANGE_ID="${3:?range is required}"
    MODE="${4:?mode is required}"
    bootstrap_python
    prepare_model_cache "$MODEL_KEY"
    nvidia-smi
    python "$EXPERIMENT_DIR/experiment.py" "$ACTION" \
      --model "$MODEL_KEY" --family "$FAMILY" --range "$RANGE_ID" --mode "$MODE"
    ;;
  cell)
    MODEL_KEY="${1:?model is required}"
    FAMILY="${2:?family is required}"
    RANGE_ID="${3:?range is required}"
    bootstrap_python
    prepare_model_cache "$MODEL_KEY"
    nvidia-smi
    python "$EXPERIMENT_DIR/experiment.py" cell \
      --model "$MODEL_KEY" --family "$FAMILY" --range "$RANGE_ID"
    ;;
  summarize)
    MODEL_KEY="${1:?model is required}"
    FAMILY="${2:?family is required}"
    python3 "$EXPERIMENT_DIR/experiment.py" summarize --model "$MODEL_KEY" --family "$FAMILY"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
