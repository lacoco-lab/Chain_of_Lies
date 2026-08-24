#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {cell|summarize} [optimization_seed]" >&2
  exit 2
fi

ACTION="$1"
shift
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT="experiments/hard_regime/multiplication_only_piggyback/diagnostics/qwen_seed2_optimization/experiment.py"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bootstrap_python() {
  python3 -m venv --system-site-packages "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir \
    'transformers==4.44.2' \
    'accelerate==0.33.0' \
    'peft==0.12.0'
}

prepare_qwen_cache() {
  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Qwen model cache check: attempt $attempt/5"
    if python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then
      echo "Model download failed; retrying after 20 seconds." >&2
      sleep 20
    fi
  done
  echo "Could not cache Qwen/Qwen2.5-7B-Instruct." >&2
  exit 1
}

case "$ACTION" in
  cell)
    if [[ $# -ne 1 ]]; then
      echo "Usage: $0 cell {3|4|5}" >&2
      exit 2
    fi
    bootstrap_python
    prepare_qwen_cache
    nvidia-smi
    exec python "$EXPERIMENT" cell --optimization-seed "$1"
    ;;
  summarize)
    exec python3 "$EXPERIMENT" summarize
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
