#!/bin/bash

set -euo pipefail

CONDOR_USER="${1:?cluster username is required}"
ACTION="${2:?action is required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/update_ablation"
DATA_ROOT="generated_data/parity_goldreich_update_ablation/seed_0"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_update_ablation"
ONE_MASK_ADAPTER="artifacts/ce_parity_goldreich_bit_attention/bit_attention/best_adapter"
BASE_ADAPTER="${ARTIFACTS_ROOT}/base/best_adapter"

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
  generate)
    python "$EXPERIMENT_DIR/generate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/validate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$DATA_ROOT"
    ;;
  prerequisite_gate)
    python "$EXPERIMENT_DIR/check_gate.py" --kind prerequisite \
      --output "$ARTIFACTS_ROOT/prerequisite_gate.json"
    ;;
  base)
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" --action base \
      --data-root "$DATA_ROOT" --output-root "$ARTIFACTS_ROOT/base" \
      --initial-adapter "$ONE_MASK_ADAPTER"
    ;;
  base_gate)
    python "$EXPERIMENT_DIR/check_gate.py" --kind base \
      --output "$ARTIFACTS_ROOT/base_gate.json"
    ;;
  derive_previous|derive_current)
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" --action "$ACTION" \
      --data-root "$DATA_ROOT" --output-root "$ARTIFACTS_ROOT/$ACTION" \
      --initial-adapter "$BASE_ADAPTER"
    ;;
  summarize)
    python "$EXPERIMENT_DIR/summarize.py" --config "$EXPERIMENT_DIR/config.json" \
      --artifacts-root "$ARTIFACTS_ROOT"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
