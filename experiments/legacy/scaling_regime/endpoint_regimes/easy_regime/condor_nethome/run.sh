#!/bin/bash
set -euo pipefail

CONDOR_USER="${1:?cluster username required}"
ACTION="${2:?action required}"
TASK="${3:-}"
MODEL="${4:-}"
SEED="${5:-}"
CONDITION="${6:-}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"

cd "$REPO_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then echo "Missing $VENV; submit experiments/cluster_jobs/setup/setup_venv.sub first." >&2; exit 2; fi
source "$VENV/bin/activate"
mkdir -p "/scratch/${CONDOR_USER}/hf_cache" "/scratch/${CONDOR_USER}/logs/chain_of_lies"
export HF_HOME="/scratch/${CONDOR_USER}/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

prepare_model_cache() {
  local model_id="$1" attempt
  if [[ "$model_id" == meta-llama/* && -z "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN is required for Llama." >&2; exit 2; fi
  for attempt in 1 2 3 4 5; do
    if python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN'))"; then return; fi
    if [[ "$attempt" != 5 ]]; then sleep 30; fi
  done
  exit 1
}

case "$ACTION" in
  generate) python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/experiment.py generate ;;
  cell)
    case "$MODEL" in qwen) MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;; llama) MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" ;; *) echo "Unknown model: $MODEL" >&2; exit 2 ;; esac
    prepare_model_cache "$MODEL_ID"; nvidia-smi
    python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/experiment.py cell --task "$TASK" --model "$MODEL" --seed "$SEED"
    ;;
  summarize) python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/summarize.py ;;
  qwen-screened-cell)
    prepare_model_cache "Qwen/Qwen2.5-7B-Instruct"; nvidia-smi
    python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/experiment.py cell --config experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/config_qwen_screened_knowledge.json --task "$TASK" --model qwen --seed "$SEED" --conditions "$CONDITION"
    ;;
  qwen-screened-summarize)
    python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/summarize.py --config experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/config_qwen_screened_knowledge.json --artifacts-root artifacts/easy_regime_qwen_screened_knowledge_v1 --tasks knowledge
    ;;
  qwen-screened-calibration-summarize)
    python experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/summarize.py --config experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/config_qwen_screened_knowledge.json --artifacts-root artifacts/easy_regime_qwen_screened_knowledge_v1 --tasks knowledge --conditions vanilla
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
