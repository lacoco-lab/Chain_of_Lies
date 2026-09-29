#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {generate|validate|cell|summarize} [model seed]" >&2
  exit 2
fi

ACTION="$1"
shift
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT="experiments/legacy/scaling_regime/endpoint_regimes/hard_regime/multiplication_only_piggyback/experiment.py"

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
  local requires_token="false"
  if [[ "$model_key" == "llama" ]]; then
    model_id="meta-llama/Llama-3.1-8B-Instruct"
    requires_token="true"
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "HF_TOKEN is required for Llama; accept its license and export the token before submission." >&2
      exit 2
    fi
  elif [[ "$model_key" == "qwen" ]]; then
    model_id="Qwen/Qwen2.5-7B-Instruct"
  else
    echo "Unknown model key: $model_key" >&2
    exit 2
  fi

  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Model cache check for $model_id: attempt $attempt/5"
    if MODEL_ID="$model_id" REQUIRES_TOKEN="$requires_token" python -c \
      "import os; from huggingface_hub import snapshot_download; token = os.environ.get('HF_TOKEN') if os.environ['REQUIRES_TOKEN'] == 'true' else None; snapshot_download(os.environ['MODEL_ID'], token=token)"; then
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
  generate)
    exec python3 "$EXPERIMENT" generate
    ;;
  validate)
    exec python3 "$EXPERIMENT" validate
    ;;
  cell)
    if [[ $# -ne 2 ]]; then
      echo "Usage: $0 cell {qwen|llama} {0|1|2}" >&2
      exit 2
    fi
    MODEL_KEY="$1"
    SEED="$2"
    bootstrap_python
    prepare_model_cache "$MODEL_KEY"
    nvidia-smi
    exec python "$EXPERIMENT" cell --model "$MODEL_KEY" --seed "$SEED"
    ;;
  summarize)
    exec python3 "$EXPERIMENT" summarize
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
