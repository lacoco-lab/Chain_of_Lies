#!/bin/bash

set -euo pipefail

CONDOR_USER="${1:?cluster username is required}"
ACTION="${2:?action is required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/final_composition"
DATA_ROOT="generated_data/parity_goldreich_final/seed_0"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_final"
MECHANISM_ADAPTER="artifacts/ce_parity_goldreich_update_ablation/base/best_adapter"
POSITIONS_ADAPTER="$ARTIFACTS_ROOT/positions/best_adapter"

cd "$REPO_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python environment: $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"

mkdir -p generated_data artifacts "/scratch/${CONDOR_USER}/hf_cache"
export HF_HOME="/scratch/${CONDOR_USER}/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

prepare_model_cache() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ.get('HF_TOKEN'))"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then sleep 30; fi
  done
  echo "Could not cache Qwen/Qwen2.5-7B-Instruct after five attempts." >&2
  exit 1
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"

case "$ACTION" in
  positions)
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" \
      --action positions --data-root "$DATA_ROOT" \
      --output-root "$ARTIFACTS_ROOT/positions" \
      --mechanism-adapter "$MECHANISM_ADAPTER"
    ;;
  full)
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" \
      --action full --data-root "$DATA_ROOT" \
      --output-root "$ARTIFACTS_ROOT/full" \
      --mechanism-adapter "$MECHANISM_ADAPTER" \
      --positions-adapter "$POSITIONS_ADAPTER"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
