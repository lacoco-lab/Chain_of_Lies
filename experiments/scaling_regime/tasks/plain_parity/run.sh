#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/scaling_regime/tasks/plain_parity"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

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
  local model_id="$1"
  if [[ "$model_id" == meta-llama/* && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for the gated Llama model." >&2
    exit 2
  fi
  echo "HF_TOKEN present: $([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo no)"
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN'))"; then
      return
    fi
    if [[ "$attempt" != 5 ]]; then sleep 30; fi
  done
  echo "Could not cache $model_id after five attempts." >&2
  exit 1
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  generate)
    CONFIG_PATH="${1:-$EXPERIMENT_DIR/config.json}"
    python "$EXPERIMENT_DIR/experiment.py" generate --config "$CONFIG_PATH"
    ;;
  cell)
    MODEL="${1:?model is required}"
    CONDITION="${2:?condition is required}"
    CONFIG_PATH="${3:-$EXPERIMENT_DIR/config.json}"
    case "$MODEL" in
      qwen) MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
      llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
      *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
    esac
    bootstrap_python
    prepare_model_cache "$MODEL_ID"
    nvidia-smi
    python "$EXPERIMENT_DIR/experiment.py" cell --config "$CONFIG_PATH" --model "$MODEL" --condition "$CONDITION"
    ;;
  summarize)
    python "$EXPERIMENT_DIR/summarize.py"
    ;;
  summarize_three)
    python "$EXPERIMENT_DIR/summarize_three_seeds.py"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
