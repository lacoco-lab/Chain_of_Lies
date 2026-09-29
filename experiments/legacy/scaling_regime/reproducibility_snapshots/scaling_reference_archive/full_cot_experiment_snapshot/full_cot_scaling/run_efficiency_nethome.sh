#!/bin/bash
set -euo pipefail
cd /nethome/mmohammadkhani/Faithfulness-Safety
source /nethome/mmohammadkhani/.venv/bin/activate
export HF_HOME=/scratch/mmohammadkhani/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ " $* " == *" --model llama "* && -z "${HF_TOKEN:-}" ]]; then
  echo 'HF_TOKEN missing from submitted environment' >&2
  exit 2
fi
exec python experiments/scaling_regime/protocols/full_cot/evaluation_efficiency.py "$@"
