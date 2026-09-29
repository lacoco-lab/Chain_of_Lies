#!/bin/bash
set -euo pipefail
cd "${_CONDOR_SCRATCH_DIR:-$PWD}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$3" == llama && -z "${HF_TOKEN:-}" ]]; then
  echo 'HF_TOKEN missing; refusing Llama job.' >&2; exit 2
fi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-cache-dir 'transformers==4.44.2' 'accelerate==0.33.0' 'peft==0.12.0'
# from_pretrained fetches needed files only, not duplicate original-format weights.
python experiments/scaling_regime/protocols/filler/recovery_sic/worker.py "$@"
