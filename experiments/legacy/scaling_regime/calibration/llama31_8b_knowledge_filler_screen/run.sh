#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
shift

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/legacy/scaling_regime/calibration/llama31_8b_knowledge_filler_screen"
DATA_ROOT="generated_data/llama31_8b_knowledge_filler"
RESPONSES_ROOT="generated_data/llama31_8b_knowledge_filler_responses"
ARTIFACTS_ROOT="artifacts/llama31_8b_knowledge_filler_screen"

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
      'accelerate==0.33.0'
    return
  fi
  source "$VENV/bin/activate"
}

prepare_model_cache() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not available inside the job." >&2
    exit 2
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    echo "Authenticated Llama cache check: attempt $attempt/5"
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', token=os.environ['HF_TOKEN'])"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then
      echo "Model download failed; retrying after 30 seconds." >&2
      sleep 30
    fi
  done
  echo "Could not cache meta-llama/Llama-3.1-8B-Instruct." >&2
  exit 1
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  prepare)
    python "$EXPERIMENT_DIR/prepare_data.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$DATA_ROOT"
    ;;

  inference)
    TASK="${1:?task is required}"
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/run_inference.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --task "$TASK" \
      --output-dir "$RESPONSES_ROOT/$TASK"
    ;;

  summarize)
    bootstrap_python
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --responses-dir "$RESPONSES_ROOT" \
      --artifacts-dir "$ARTIFACTS_ROOT"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
