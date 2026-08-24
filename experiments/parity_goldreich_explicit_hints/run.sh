#!/bin/bash

set -euo pipefail

ACTION="${1:?action is required}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/parity_goldreich_explicit_hints"
DATA_ROOT="generated_data/parity_goldreich_explicit_hints/seed_0"
DATA_ARCHIVE="generated_data/parity_goldreich_explicit_hints_seed_0.tar.gz"
RESPONSES_ROOT="generated_data/parity_goldreich_explicit_hints_responses/seed_0"
CORE_RESPONSES_ARCHIVE="generated_data/parity_goldreich_explicit_hints_core_responses.tar.gz"
EXTENSION_RESPONSES_ARCHIVE="generated_data/parity_goldreich_explicit_hints_extension_responses.tar.gz"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_explicit_hints"
BASE_ADAPTER="artifacts/ce_parity_goldreich_curriculum/update/final_adapter"

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
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ.get('HF_TOKEN'))"; then
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
echo "HF_TOKEN present: $([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo no; true)"

case "$ACTION" in
  generate)
    python "$EXPERIMENT_DIR/generate_data.py" --config "$EXPERIMENT_DIR/config.json" --output-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/validate_data.py" --config "$EXPERIMENT_DIR/config.json" --data-root "$DATA_ROOT"
    tar -czf "$DATA_ARCHIVE" "$DATA_ROOT"
    rm -rf "$DATA_ROOT"
    ;;
  core)
    unpack_data
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/run_track.py" \
      --config "$EXPERIMENT_DIR/config.json" --track core --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" --responses-root "$RESPONSES_ROOT" \
      --initial-adapter "$BASE_ADAPTER"
    tar -czf "$CORE_RESPONSES_ARCHIVE" "$RESPONSES_ROOT/core"
    rm -rf "$RESPONSES_ROOT/core"
    ;;
  extension)
    unpack_data
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/run_track.py" \
      --config "$EXPERIMENT_DIR/config.json" --track extension --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" --responses-root "$RESPONSES_ROOT" \
      --initial-adapter "$ARTIFACTS_ROOT/core/final_adapter"
    tar -czf "$EXTENSION_RESPONSES_ARCHIVE" "$RESPONSES_ROOT/extension"
    rm -rf "$RESPONSES_ROOT/extension"
    ;;
  summarize)
    unpack_data
    tar -xzf "$CORE_RESPONSES_ARCHIVE"
    tar -xzf "$EXTENSION_RESPONSES_ARCHIVE"
    python "$EXPERIMENT_DIR/summarize.py" \
      --config "$EXPERIMENT_DIR/config.json" --data-root "$DATA_ROOT" \
      --artifacts-root "$ARTIFACTS_ROOT" --responses-root "$RESPONSES_ROOT"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
