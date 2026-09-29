#!/bin/bash

# Run one manually submitted stage on the conduit2 Docker pool. Each stage
# checks that the outputs required from the preceding stage are present.
set -euo pipefail

ACTION="${1:?action is required}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/final_composition"
DATA_ROOT="generated_data/parity_goldreich_final/seed_0"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_final"
MECHANISM_ROOT="artifacts/ce_parity_goldreich_update_ablation/base"
MECHANISM_ADAPTER="$MECHANISM_ROOT/best_adapter"
POSITIONS_ROOT="$ARTIFACTS_ROOT/positions"
POSITIONS_ADAPTER="$POSITIONS_ROOT/best_adapter"

cd "$JOB_ROOT"
mkdir -p generated_data artifacts hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing prerequisite file: $1" >&2
    exit 2
  fi
}

bootstrap_python() {
  python3 -m venv --system-site-packages "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir \
    'transformers==4.57.6' \
    'accelerate==1.10.1' \
    'peft==0.17.1'
  python -c "import accelerate, peft, torch, transformers; assert torch.__version__.split('+')[0] == '2.8.0'; assert transformers.__version__ == '4.57.6'; assert accelerate.__version__ == '1.10.1'; assert peft.__version__ == '0.17.1'; print('Runtime:', 'torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'peft', peft.__version__)"
}

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
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  prepare)
    require_file "$MECHANISM_ROOT/metrics.json"
    require_file "$MECHANISM_ADAPTER/adapter_model.safetensors"
    python "$EXPERIMENT_DIR/generate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/validate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/check_gate.py" --kind mechanism \
      --output "$ARTIFACTS_ROOT/mechanism_gate.json"
    ;;
  positions)
    # Ensure Condor can return the declared output directory even if setup or
    # training fails, preserving the real process error instead of masking it.
    mkdir -p "$POSITIONS_ROOT"
    require_file "$DATA_ROOT/manifest.json"
    require_file "$ARTIFACTS_ROOT/mechanism_gate.json"
    require_file "$MECHANISM_ADAPTER/adapter_model.safetensors"
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" \
      --action positions --data-root "$DATA_ROOT" \
      --output-root "$POSITIONS_ROOT" \
      --mechanism-adapter "$MECHANISM_ADAPTER"
    ;;
  positions_gate)
    python "$EXPERIMENT_DIR/check_gate.py" --kind positions \
      --output "$ARTIFACTS_ROOT/positions_gate.json"
    ;;
  full)
    mkdir -p "$ARTIFACTS_ROOT/full"
    require_file "$DATA_ROOT/manifest.json"
    require_file "$ARTIFACTS_ROOT/positions_gate.json"
    require_file "$MECHANISM_ADAPTER/adapter_model.safetensors"
    require_file "$POSITIONS_ADAPTER/adapter_model.safetensors"
    bootstrap_python
    prepare_model_cache
    nvidia-smi
    python "$EXPERIMENT_DIR/train.py" --config "$EXPERIMENT_DIR/config.json" \
      --action full --data-root "$DATA_ROOT" \
      --output-root "$ARTIFACTS_ROOT/full" \
      --mechanism-adapter "$MECHANISM_ADAPTER" \
      --positions-adapter "$POSITIONS_ADAPTER"
    ;;
  summarize)
    require_file "$POSITIONS_ROOT/metrics.json"
    require_file "$ARTIFACTS_ROOT/full/metrics.json"
    python "$EXPERIMENT_DIR/summarize.py" --config "$EXPERIMENT_DIR/config.json" \
      --artifacts-root "$ARTIFACTS_ROOT"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
