#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/exploratory/curriculum"
DATA_ROOT="generated_data/parity_goldreich_curriculum/seed_0"
DATA_ARCHIVE="generated_data/parity_goldreich_curriculum_seed_0.tar.gz"
RESPONSES_ROOT="generated_data/parity_goldreich_curriculum_responses/seed_0"
UPDATE_RESPONSES_ARCHIVE="generated_data/parity_goldreich_curriculum_update_responses.tar.gz"
GOLDREICH_RESPONSES_ARCHIVE="generated_data/parity_goldreich_curriculum_goldreich_responses.tar.gz"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_curriculum"

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
    echo "HF_TOKEN is not available inside the GPU job." >&2
    exit 2
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ['HF_TOKEN'])"; then
      return
    fi
    if [[ "$attempt" != "5" ]]; then
      sleep 30
    fi
  done
  echo "Could not cache the model after five attempts." >&2
  exit 1
}

unpack_data() {
  if [[ ! -d "$DATA_ROOT" ]]; then
    tar -xzf "$DATA_ARCHIVE"
  fi
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  generate)
    python "$EXPERIMENT_DIR/generate_data.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/validate_data.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$DATA_ROOT"
    tar -czf "$DATA_ARCHIVE" "$DATA_ROOT"
    rm -rf "$DATA_ROOT"
    ;;

  update)
    unpack_data
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/run_track.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --track update \
      --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" \
      --responses-root "$RESPONSES_ROOT"
    tar -czf "$UPDATE_RESPONSES_ARCHIVE" "$RESPONSES_ROOT/update"
    rm -rf "$RESPONSES_ROOT/update"
    ;;

  goldreich)
    unpack_data
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/run_track.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --track goldreich \
      --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" \
      --responses-root "$RESPONSES_ROOT" \
      --initial-adapter "$ARTIFACTS_ROOT/update/final_adapter"
    tar -czf "$GOLDREICH_RESPONSES_ARCHIVE" "$RESPONSES_ROOT/goldreich"
    rm -rf "$RESPONSES_ROOT/goldreich"
    ;;

  summarize)
    unpack_data
    tar -xzf "$UPDATE_RESPONSES_ARCHIVE"
    tar -xzf "$GOLDREICH_RESPONSES_ARCHIVE"
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" \
      --responses-root "$RESPONSES_ROOT"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
