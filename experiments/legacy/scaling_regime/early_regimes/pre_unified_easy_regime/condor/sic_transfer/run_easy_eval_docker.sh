#!/bin/bash

set -euo pipefail

ACTION="${1:-runtime_smoke}"
VARIANT="${2:-}"
MODE="${3:-}"
SEED="${4:-0}"

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
VENV="$JOB_ROOT/.venv"
SPLIT_ROOT="generated_data/regime_splits"
ARTIFACTS_ROOT="artifacts/ce_easy_regime"

cd "$JOB_ROOT"
mkdir -p artifacts generated_data hf_cache
export HF_HOME="$JOB_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$JOB_ROOT/hf_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create every declared output before dependency setup so Condor can return the
# real process status if installation or inference fails.
if [[ "$ACTION" == "runtime_smoke" ]]; then
  mkdir -p artifacts/easy_runtime_smoke
elif [[ "$ACTION" == "eval" ]]; then
  if [[ -z "$VARIANT" || -z "$MODE" ]]; then
    echo "eval requires variant and mode" >&2
    exit 2
  fi
  ROOT="$ARTIFACTS_ROOT/seed_$SEED/$MODE"
  ADAPTER="$ROOT/$VARIANT"
  RESPONSES="generated_data/regime_eval_responses/seed_$SEED/$MODE/$VARIANT"
  mkdir -p "$ADAPTER/per_variant_eval/ckpt_final" "$RESPONSES"
fi

python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --no-cache-dir \
  'matplotlib>=3.8' \
  'transformers==4.44.2' \
  'accelerate==0.33.0' \
  'peft==0.12.0'

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

case "$ACTION" in
  runtime_smoke)
    nvidia-smi
    python -c "import torch, transformers, peft; assert torch.cuda.is_available(); print('runtime smoke passed:', torch.cuda.get_device_name(0))"
    printf 'runtime smoke passed\n' > artifacts/easy_runtime_smoke/runtime_ok.txt
    ;;

  eval)
    nvidia-smi
    python scripts/shared/evaluation/evaluate_ce.py \
      --artifacts-root "$ROOT" \
      --variant "$VARIANT" \
      --eval-responses-root "$RESPONSES" \
      --model-responses-subdir finetuned \
      --checkpoint ckpt_final \
      --max-new-tokens 512 \
      --temperature 0.0 \
      --greedy \
      --inference-batch-size 4 \
      --no-resume \
      --json-out "$ADAPTER/per_variant_eval/ckpt_final/$VARIANT.json"

    if [[ "$VARIANT" == "knowledge_easy_1fact" ]]; then
      python scripts/shared/evaluation/evaluate_ce.py \
        --artifacts-root "$ROOT" \
        --variant "$VARIANT" \
        --prompts-dir "$SPLIT_ROOT/seed_$SEED/$VARIANT/component_val_prompts" \
        --eval-responses-root "$RESPONSES/components" \
        --model-responses-subdir finetuned \
        --checkpoint ckpt_final \
        --max-new-tokens 256 \
        --temperature 0.0 \
        --greedy \
        --inference-batch-size 4 \
        --no-resume \
        --json-out "$ADAPTER/per_variant_eval/ckpt_final/knowledge_easy_components.json"
    fi
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
