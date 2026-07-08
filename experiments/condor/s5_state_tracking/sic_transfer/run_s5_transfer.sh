#!/bin/bash
# SIC/conduit fallback workflow that avoids user-writable shared scratch.
# Runs inside /scratch/common/images/apptainer-builder.sif, builds a temporary
# PyTorch sandbox dir in the Condor execute directory, then runs the S5 command
# inside it. A sandbox directory is used instead of a .sif because mounting a
# SquashFS .sif needs FUSE (squashfuse), which is unavailable on the execute
# nodes when running Apptainer nested inside the builder container.

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
  echo "Neither apptainer nor singularity is available inside the builder image." >&2
  exit 1
fi

echo "Host: $HOSTNAME"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"
echo "Outer runtime: $APPTAINER"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -d "$PYTORCH_DIR" ]]; then
  echo "Building temporary PyTorch sandbox at $PYTORCH_DIR"
  "$APPTAINER" build --sandbox "$PYTORCH_DIR" "$SOURCE_IMAGE"
fi

run_in_pytorch() {
  "$APPTAINER" exec --nv \
    --env HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$PYTORCH_DIR" bash -lc "$*"
}

# Pinned, mutually compatible versions known to work with torch 2.3.1.
# Pinned here (not in requirements.txt) so the other clusters keep their own
# resolution. transformers is pinned because newer releases hit
# "NameError: name 'nn' is not defined" on import.
PINNED_PACKAGES="transformers==4.44.2 accelerate==0.33.0 peft==0.12.0"

bootstrap_python() {
  run_in_pytorch "python3 -m venv --system-site-packages '$VENV' && \
    source '$VENV/bin/activate' && \
    python -m pip install --no-cache-dir 'matplotlib>=3.8' $PINNED_PACKAGES && \
    python - <<'PY'
import torch
import transformers
import accelerate
import peft
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('peft', peft.__version__)
PY"
}

case "$ACTION" in
  setup)
    bootstrap_python
    ;;

  gen_splits)
    SEED="${1:-0}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python scripts/generate_prompt_splits.py \
        --variant all_s5 \
        --train-n 10000 \
        --val-n 1000 \
        --seed '$SEED' \
        --output-root 'generated_data/prompt_splits/seed_$SEED' && \
      python scripts/sanity_check_s5.py --split-root 'generated_data/prompt_splits/seed_$SEED' --variant s5_piggyback && \
      python scripts/sanity_check_s5.py --split-root 'generated_data/prompt_splits/seed_$SEED' --variant s5_control"
    ;;

  train)
    VARIANT_NAME="${1:-s5_piggyback}"
    SUPERVISION_MODE="${2:-public_cot}"
    SEED="${3:-0}"
    EPOCHS="${4:-1}"
    SAVE_EVERY="${5:-5000}"
    EVAL_EVERY="${6:-5000}"
    ARTIFACTS_ROOT="${7:-artifacts/ce_s5}"
    MAX_NEW_TOKENS="${8:-512}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      python scripts/run_ce_only.py \
        --variant '$VARIANT_NAME' \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-root '$ARTIFACTS_ROOT/seed_$SEED/$SUPERVISION_MODE' \
        --split-root 'generated_data/prompt_splits/seed_$SEED' \
        --epochs '$EPOCHS' \
        --batch-size 1 \
        --learning-rate 2e-5 \
        --max-new-tokens '$MAX_NEW_TOKENS' \
        --save-every '$SAVE_EVERY' \
        --eval-every '$EVAL_EVERY' \
        --validation-sample-size 200 \
        --validation-batch-size 1 \
        --expected-train-prompts 10000 \
        --expected-val-prompts 1000 \
        --supervision-mode '$SUPERVISION_MODE' \
        --seed '$SEED'"
    ;;

  eval)
    MODE="${1:-public_cot}"
    VARIANT_NAME="${2:-s5_piggyback}"
    SEED="${3:-0}"
    ARTIFACTS_ROOT="${4:-artifacts/ce_s5}"
    MAX_NEW_TOKENS="${5:-512}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      nvidia-smi || true && \
      ARTIFACT_ROOT='$ARTIFACTS_ROOT/seed_$SEED/$MODE' && \
      RESPONSES_ROOT='generated_data/eval_responses/'\"\$(basename '$ARTIFACTS_ROOT')\"'/seed_$SEED/$MODE' && \
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
    ARTIFACTS_ROOT="${1:-artifacts/ce_s5}"
    bootstrap_python
    run_in_pytorch "source '$VENV/bin/activate' && \
      python scripts/summarize_ce_results.py \
        --root '$ARTIFACTS_ROOT' \
        --modes answer_only,public_cot,verbose_public_cot \
        --variants s5_piggyback,s5_control \
        --piggyback-variant s5_piggyback \
        --control-variant s5_control"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
