#!/bin/bash

set -euo pipefail

CONDOR_USER="${1:?cluster username is required}"
ACTION="${2:?action is required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/exploratory/encrypted_bridge"
CURRICULUM_DIR="experiments/goldreich_parity/qwen_attempts/exploratory/curriculum"
BIT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/bit_attention"

CURRICULUM_DATA="generated_data/parity_goldreich_curriculum/seed_0"
CURRICULUM_RESPONSES="generated_data/parity_goldreich_curriculum_responses/seed_0"
CURRICULUM_ARTIFACTS="artifacts/ce_parity_goldreich_curriculum"

BIT_DATA="generated_data/parity_goldreich_bit_attention/seed_0"
BIT_ARTIFACTS="artifacts/ce_parity_goldreich_bit_attention"

BRIDGE_DATA="generated_data/parity_goldreich_encrypted_bridge/seed_0"
BRIDGE_ARTIFACTS="artifacts/ce_parity_goldreich_encrypted_bridge"
BRIDGE_TRAINING="${BRIDGE_ARTIFACTS}/training"

cd "$REPO_ROOT"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python environment: $VENV" >&2
  echo "Run: condor_submit experiments/cluster_jobs/setup/setup_venv.sub" >&2
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
echo "Repository: $REPO_ROOT"

case "$ACTION" in
  curriculum_generate)
    python "$CURRICULUM_DIR/generate_data.py" \
      --config "$CURRICULUM_DIR/config.json" \
      --output-root "$CURRICULUM_DATA"
    python "$CURRICULUM_DIR/validate_data.py" \
      --config "$CURRICULUM_DIR/config.json" \
      --data-root "$CURRICULUM_DATA"
    ;;

  bit_generate)
    python "$BIT_DIR/generate_data.py" \
      --config "$BIT_DIR/config.json" \
      --output-root "$BIT_DATA"
    python "$BIT_DIR/validate_data.py" \
      --config "$BIT_DIR/config.json" \
      --data-root "$BIT_DATA"
    ;;

  bridge_generate)
    python "$EXPERIMENT_DIR/generate_data.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$BRIDGE_DATA"
    python "$EXPERIMENT_DIR/validate_data.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$BRIDGE_DATA"
    ;;

  curriculum_update)
    prepare_model_cache
    nvidia-smi
    python "$CURRICULUM_DIR/run_track.py" \
      --config "$CURRICULUM_DIR/config.json" \
      --track update \
      --data-root "$CURRICULUM_DATA" \
      --artifacts-root "$CURRICULUM_ARTIFACTS" \
      --responses-root "$CURRICULUM_RESPONSES"
    ;;

  bit_attention)
    prepare_model_cache
    nvidia-smi
    python "$BIT_DIR/train_condition.py" \
      --config "$BIT_DIR/config.json" \
      --condition bit_attention \
      --data-root "$BIT_DATA" \
      --artifacts-root "$BIT_ARTIFACTS" \
      --initial-adapter "$CURRICULUM_ARTIFACTS/update/final_adapter"
    ;;

  prerequisite_gate)
    python "$EXPERIMENT_DIR/check_prerequisite.py"
    ;;

  bridge_train)
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train_bridge.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$BRIDGE_DATA" \
      --output-root "$BRIDGE_TRAINING" \
      --initial-adapter "$BIT_ARTIFACTS/bit_attention/best_adapter"
    ;;

  bridge_summarize)
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --training-root "$BRIDGE_TRAINING" \
      --output-root "$BRIDGE_ARTIFACTS"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

