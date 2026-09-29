#!/bin/bash

set -euo pipefail

ACTION="${1:-setup}"
shift || true

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
PYTORCH_DIR="$JOB_ROOT/pytorch_sandbox"
SOURCE_IMAGE="docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
VENV="$JOB_ROOT/.venv"
CONFIG="experiments/legacy/scaling_regime/early_regimes/pre_unified_easy_regime/easy_regime.json"
SPLIT_ROOT="generated_data/regime_splits"
ARTIFACTS_ROOT="artifacts/ce_easy_regime"

mkdir -p "$JOB_ROOT/.apptainer/cache" "$JOB_ROOT/.apptainer/tmp" "$JOB_ROOT/hf_cache"
export APPTAINER_CACHEDIR="$JOB_ROOT/.apptainer/cache"
export SINGULARITY_CACHEDIR="$JOB_ROOT/.apptainer/cache"
export APPTAINER_TMPDIR="$JOB_ROOT/.apptainer/tmp"
export SINGULARITY_TMPDIR="$JOB_ROOT/.apptainer/tmp"
export HF_HOME="$JOB_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$JOB_ROOT/hf_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER=singularity
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 1
fi

cd "$JOB_ROOT"
mkdir -p generated_data artifacts

if [[ ! -d "$PYTORCH_DIR" ]]; then
  "$APPTAINER" build --sandbox "$PYTORCH_DIR" "$SOURCE_IMAGE"
fi

run_in_pytorch() {
  "$APPTAINER" exec \
    --env HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$PYTORCH_DIR" bash -lc "$*"
}

run_in_pytorch_gpu() {
  "$APPTAINER" exec --nv \
    --env HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$PYTORCH_DIR" bash -lc "$*"
}

PINNED_PACKAGES="transformers==4.44.2 accelerate==0.33.0 peft==0.12.0"

bootstrap_python() {
  run_in_pytorch "python3 -m venv --system-site-packages '$VENV' && \
    source '$VENV/bin/activate' && \
    python -m pip install --no-cache-dir 'matplotlib>=3.8' $PINNED_PACKAGES"
}

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"

case "$ACTION" in
  gen_splits)
    SEED="${1:-0}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python experiments/legacy/scaling_regime/early_regimes/pre_unified_easy_regime/generate_regime_splits.py \
        --config '$CONFIG' \
        --seed '$SEED' \
        --output-root '$SPLIT_ROOT'"
    ;;

  train)
    VARIANT="${1:?variant is required}"
    MODE="${2:?supervision mode is required}"
    SEED="${3:-0}"
    EPOCHS="${4:-1}"
    bootstrap_python
    run_in_pytorch_gpu "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      python scripts/shared/training/run_ce.py \
        --variant '$VARIANT' \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-root '$ARTIFACTS_ROOT/seed_$SEED/$MODE' \
        --split-root '$SPLIT_ROOT/seed_$SEED' \
        --epochs '$EPOCHS' \
        --batch-size 1 \
        --learning-rate 2e-5 \
        --max-new-tokens 512 \
        --save-every 10000 \
        --eval-every 2500 \
        --validation-sample-size 200 \
        --validation-batch-size 2 \
        --expected-train-prompts 10000 \
        --expected-val-prompts 1000 \
        --supervision-mode '$MODE' \
        --seed '$SEED'"
    ;;

  eval)
    VARIANT="${1:?variant is required}"
    MODE="${2:?supervision mode is required}"
    SEED="${3:-0}"
    ROOT="$ARTIFACTS_ROOT/seed_$SEED/$MODE"
    ADAPTER="$ROOT/$VARIANT"
    RESPONSES="generated_data/regime_eval_responses/seed_$SEED/$MODE/$VARIANT"
    # Create both declared Condor output roots outside the nested runtime.  If
    # image startup or Python fails, Condor can still transfer diagnostics and
    # report the real process exit instead of replacing it with hold code 12/2.
    mkdir -p "$ADAPTER/per_variant_eval/ckpt_final" "$RESPONSES"
    bootstrap_python
    run_in_pytorch_gpu "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      python scripts/shared/evaluation/evaluate_ce.py \
        --artifacts-root '$ROOT' \
        --variant '$VARIANT' \
        --eval-responses-root '$RESPONSES' \
        --model-responses-subdir finetuned \
        --checkpoint ckpt_final \
        --max-new-tokens 512 \
        --temperature 0.0 \
        --greedy \
        --inference-batch-size 4 \
        --no-resume \
        --json-out '$ADAPTER/per_variant_eval/ckpt_final/$VARIANT.json' && \
      if [[ '$VARIANT' == 'knowledge_easy_1fact' ]]; then \
        python scripts/shared/evaluation/evaluate_ce.py \
          --artifacts-root \"\$ROOT\" \
          --variant '$VARIANT' \
          --prompts-dir '$SPLIT_ROOT/seed_$SEED/$VARIANT/component_val_prompts' \
          --eval-responses-root '$RESPONSES/components' \
          --model-responses-subdir finetuned \
          --checkpoint ckpt_final \
          --max-new-tokens 256 \
          --temperature 0.0 \
          --greedy \
          --inference-batch-size 4 \
          --no-resume \
          --json-out '$ADAPTER/per_variant_eval/ckpt_final/knowledge_easy_components.json'; \
      fi"
    ;;

  summarize)
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python experiments/legacy/scaling_regime/early_regimes/pre_unified_easy_regime/summarize_regime_results.py \
        --root '$ARTIFACTS_ROOT' \
        --checkpoint ckpt_final \
        --modes answer_only,public_cot \
        --variants mul_easy,s5_easy,knowledge_easy_1fact,knowledge_easy_components"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
