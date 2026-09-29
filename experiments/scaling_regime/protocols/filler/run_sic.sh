#!/bin/bash
set -euo pipefail

TASK="${1:?task required}"
MODEL="${2:?model required}"
SEED="${3:?seed required}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"

cd "$JOB_ROOT"
mkdir -p artifacts generated_data hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "$TASK" in multiplication|knowledge|s5|parity) ;; *) echo "Unsupported SIC task: $TASK" >&2; exit 2 ;; esac
case "$MODEL" in
  qwen) MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
  llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
if [[ "$MODEL" == llama && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for Llama-3.1." >&2
  exit 2
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv --system-site-packages "$VENV"
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir \
    'transformers==4.44.2' 'accelerate==0.33.0' 'peft==0.12.0'
else
  source "$VENV/bin/activate"
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

PREFLIGHT_REPORT="artifacts/filler_only_scaling_v1/${TASK}/${MODEL}/seed_${SEED}/filler_only/provenance/preflight.json"
python experiments/scaling_regime/protocols/filler/preflight.py \
  --task "$TASK" --model "$MODEL" --seed "$SEED" --report "$PREFLIGHT_REPORT"

nvidia-smi
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'architectures': torch.cuda.get_arch_list()}, flush=True); print(torch.zeros(1, device='cuda'), flush=True)"
python experiments/scaling_regime/protocols/filler/experiment.py cell \
  --task "$TASK" --model "$MODEL" --seed "$SEED"
