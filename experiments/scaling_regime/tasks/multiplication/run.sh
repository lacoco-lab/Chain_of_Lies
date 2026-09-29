#!/bin/bash
set -euo pipefail

ACTION="${1:?action required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT="experiments/scaling_regime/tasks/multiplication"

cd "$JOB_ROOT"
mkdir -p artifacts generated_data hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
  else
    source "$VENV/bin/activate"
  fi
}

cache_model() {
  local model_key="$1"
  local model_id
  case "$model_key" in
    qwen) model_id='Qwen/Qwen2.5-7B-Instruct' ;;
    llama) model_id='meta-llama/Llama-3.1-8B-Instruct' ;;
    *) echo "Unknown model: $model_key" >&2; exit 2 ;;
  esac
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN present: yes"
  else
    echo "HF_TOKEN present: no"
  fi
  if [[ "$model_key" == llama && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for Llama-3.1." >&2
    exit 2
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN'))"; then
      return
    fi
    if [[ "$attempt" == 5 ]]; then
      echo "Could not cache ${model_id} after five attempts." >&2
      exit 1
    fi
    sleep 30
  done
}

case "$ACTION" in
  generate)
    python "$EXPERIMENT/experiment.py" generate
    ;;
  cell)
    MODEL="${1:?model required}"
    SEED="${2:?seed required}"
    CONDITION="${3:?condition required}"
    bootstrap_python
    cache_model "$MODEL"
    nvidia-smi
    python "$EXPERIMENT/experiment.py" cell \
      --model "$MODEL" --seed "$SEED" --condition "$CONDITION"
    ;;
  summarize)
    bootstrap_python
    python "$EXPERIMENT/summarize.py"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
