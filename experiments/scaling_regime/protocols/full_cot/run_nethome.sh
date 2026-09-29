#!/bin/bash
set -euo pipefail
TASK="${1:?task required}"
MODEL="${2:-qwen}"
SEED="${3:-0}"
ACTION="${4:-cell}"
cd /nethome/mmohammadkhani/Faithfulness-Safety
if [[ "$ACTION" == check ]]; then
  exec /nethome/mmohammadkhani/.venv/bin/python -S experiments/scaling_regime/protocols/full_cot/experiment.py check --task "$TASK"
fi
source /nethome/mmohammadkhani/.venv/bin/activate
export HF_HOME=/scratch/mmohammadkhani/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODEL" == llama && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN missing from submitted job environment" >&2
  exit 2
fi
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True); print(torch.zeros(1, device='cuda'), flush=True)"
exec python experiments/scaling_regime/protocols/full_cot/experiment.py cell --task "$TASK" --model "$MODEL" --seed "$SEED"

