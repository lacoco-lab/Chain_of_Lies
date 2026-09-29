#!/bin/bash
set -euo pipefail

CONDOR_USER="${1:?cluster username required}"
MODEL="${2:?model required}"
SEED="${3:?seed required}"
CONDITION="${4:?condition required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"

cd "$REPO_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python environment: $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"
mkdir -p "/scratch/${CONDOR_USER}/hf_cache" "/scratch/${CONDOR_USER}/logs/chain_of_lies"
export HF_HOME="/scratch/${CONDOR_USER}/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "$MODEL" in
  qwen) MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
  llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
if [[ "$MODEL" == "llama" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for Llama-3.1." >&2
  exit 2
fi

for attempt in 1 2 3 4 5; do
  if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', token=os.environ.get('HF_TOKEN'))"; then
    break
  fi
  if [[ "$attempt" == 5 ]]; then
    echo "Could not cache ${MODEL_ID} after five attempts." >&2
    exit 1
  fi
  sleep 30
done

nvidia-smi
python experiments/scaling_regime/tasks/s5_state_tracking/experiment.py cell \
  --model "$MODEL" --seed "$SEED" --condition "$CONDITION"

