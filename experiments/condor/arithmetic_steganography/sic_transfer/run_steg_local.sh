#!/bin/bash
# SIC transfer workflow for aligned local-trace steganography CE experiments.

set -euo pipefail

ACTION="${1:-setup}"
shift || true

JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
PYTORCH_DIR="$JOB_ROOT/pytorch_sandbox"
SOURCE_IMAGE="docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
VENV="$JOB_ROOT/.venv"

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

bootstrap_python() {
  run_in_pytorch "python3 -m venv --system-site-packages '$VENV' && \
    source '$VENV/bin/activate' && \
    python -m pip install --no-cache-dir 'matplotlib>=3.8' \
      transformers==4.44.2 accelerate==0.33.0 peft==0.12.0"
}

case "$ACTION" in
  setup)
    bootstrap_python
    ;;

  gen_splits)
    SEED="${1:-0}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python scripts/generate_steganography_splits.py \
        --train-n 10000 \
        --val-n 1000 \
        --seed '$SEED' \
        --output-root 'generated_data/prompt_splits/seed_$SEED' \
        --variants arith_steg_local_invisible && \
      python scripts/sanity_check_steganography.py \
        --split-root 'generated_data/prompt_splits/seed_$SEED' \
        --expected-train 10000 \
        --expected-val 1000 \
        --variants arith_steg_local_invisible && \
      python scripts/check_steganography_tokenizer.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --split-root 'generated_data/prompt_splits/seed_$SEED' \
        --sample-size 20"
    ;;

  train)
    VARIANT_NAME="${1:?variant required}"
    SEED="${2:-0}"
    MAX_NEW_TOKENS="${3:-1536}"
    bootstrap_python
    run_in_pytorch_gpu "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      python scripts/run_ce_only.py \
        --variant '$VARIANT_NAME' \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-root 'artifacts/ce_steganography_local/seed_$SEED/local_channel_cot' \
        --split-root 'generated_data/prompt_splits/seed_$SEED' \
        --epochs 1 \
        --batch-size 1 \
        --learning-rate 2e-5 \
        --max-new-tokens '$MAX_NEW_TOKENS' \
        --save-every 10000 \
        --eval-every 10000 \
        --validation-sample-size 200 \
        --validation-batch-size 1 \
        --expected-train-prompts 10000 \
        --expected-val-prompts 1000 \
        --supervision-mode local_channel_cot \
        --seed '$SEED'"
    ;;

  eval)
    VARIANT_NAME="${1:?variant required}"
    SEED="${2:-0}"
    MAX_NEW_TOKENS="${3:-2048}"
    bootstrap_python
    run_in_pytorch_gpu "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      ARTIFACT_ROOT='artifacts/ce_steganography_local/seed_$SEED/local_channel_cot' && \
      RESPONSES_ROOT='generated_data/eval_responses/ce_steganography_local/seed_$SEED/local_channel_cot' && \
      mkdir -p \"\$ARTIFACT_ROOT/per_variant_eval/ckpt_task\" && \
      python scripts/evaluate_ce_model.py \
        --artifacts-root \"\$ARTIFACT_ROOT\" \
        --variant '$VARIANT_NAME' \
        --eval-responses-root \"\$RESPONSES_ROOT\" \
        --model-responses-subdir trained \
        --checkpoint ckpt_task \
        --max-new-tokens '$MAX_NEW_TOKENS' \
        --temperature 0.0 \
        --greedy \
        --inference-batch-size 2 \
        --no-resume \
        --json-out \"\$ARTIFACT_ROOT/per_variant_eval/ckpt_task/$VARIANT_NAME.json\""
    ;;

  summarize)
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python scripts/summarize_ce_results.py \
        --root artifacts/ce_steganography_local \
        --modes local_channel_cot \
        --variants arith_steg_local_invisible \
        --skip-pair-deltas"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
