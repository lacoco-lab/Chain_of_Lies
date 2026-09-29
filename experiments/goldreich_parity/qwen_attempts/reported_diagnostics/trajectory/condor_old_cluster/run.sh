#!/bin/bash

set -euo pipefail

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/trajectory"
DATA_ROOT="generated_data/parity_goldreich_trajectory/seed_0"
PREREQUISITE_DATA="generated_data/parity_goldreich_final/seed_0"
OUTPUT_ROOT="artifacts/ce_parity_goldreich_trajectory"
LENGTH="${1:?Usage: run.sh LENGTH}"
OUTPUT_ROOT="$OUTPUT_ROOT/n$LENGTH"

cd "$JOB_ROOT"
mkdir -p "$OUTPUT_ROOT" hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --no-cache-dir \
  'transformers==4.57.6' \
  'accelerate==1.10.1' \
  'peft==0.17.1'
python -c "import accelerate, peft, torch, transformers; assert torch.__version__.split('+')[0] == '2.8.0'; assert transformers.__version__ == '4.57.6'; assert accelerate.__version__ == '1.10.1'; assert peft.__version__ == '0.17.1'; print('Runtime:', torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__)"

for attempt in 1 2 3 4 5; do
  if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ.get('HF_TOKEN'))"; then
    break
  fi
  if [[ "$attempt" == "5" ]]; then exit 1; fi
  sleep 30
done

nvidia-smi
python "$EXPERIMENT_DIR/train.py" \
  --config "$EXPERIMENT_DIR/config.json" \
  --data-root "$DATA_ROOT" \
  --prerequisite-data-root "$PREREQUISITE_DATA" \
  --output-root "$OUTPUT_ROOT" \
  --length "$LENGTH" \
  --adapter artifacts/ce_parity_goldreich_update_ablation/base/best_adapter \
  --adapter artifacts/ce_parity_goldreich_final/positions/best_adapter \
  --adapter artifacts/ce_parity_goldreich_final/full/best_adapter
