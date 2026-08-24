#!/bin/bash
# SIC transfer workflow for the inference-only preliminary filler gate.

set -euo pipefail

ACTION="${1:-run}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
PYTORCH_DIR="$JOB_ROOT/pytorch_sandbox"
SOURCE_IMAGE="docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
VENV="$JOB_ROOT/.venv"

CONFIG_PATH="preliminary_filler_gate/config/default.json"
RUN_DIR="preliminary_filler_gate/runs/qwen25_7b_default"

mkdir -p "$JOB_ROOT/.apptainer/cache" "$JOB_ROOT/.apptainer/tmp" "$JOB_ROOT/hf_cache"
export APPTAINER_CACHEDIR="$JOB_ROOT/.apptainer/cache"
export SINGULARITY_CACHEDIR="$JOB_ROOT/.apptainer/cache"
export APPTAINER_TMPDIR="$JOB_ROOT/.apptainer/tmp"
export SINGULARITY_TMPDIR="$JOB_ROOT/.apptainer/tmp"
export HF_HOME="$JOB_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$JOB_ROOT/hf_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER=singularity
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 1
fi

cd "$JOB_ROOT"

echo "Host: ${HOSTNAME:-unknown}"
echo "Action: $ACTION"
echo "Job root: $JOB_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -d "$PYTORCH_DIR" ]]; then
  echo "Building temporary PyTorch sandbox at $PYTORCH_DIR"
  "$APPTAINER" build --sandbox "$PYTORCH_DIR" "$SOURCE_IMAGE"
fi

run_in_pytorch() {
  "$APPTAINER" exec \
    --env HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF",PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false \
    "$PYTORCH_DIR" bash -lc "$*"
}

run_in_pytorch_gpu() {
  "$APPTAINER" exec --nv \
    --env HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF",PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false \
    "$PYTORCH_DIR" bash -lc "$*"
}

bootstrap_python() {
  run_in_pytorch "python3 -m venv --system-site-packages '$VENV' && \
    source '$VENV/bin/activate' && \
    python -m pip install --no-cache-dir \
      'matplotlib>=3.8' \
      transformers==4.44.2 \
      accelerate==0.33.0"
}

case "$ACTION" in
  run)
    bootstrap_python
    run_in_pytorch_gpu "set -euo pipefail && \
      source '$VENV/bin/activate' && \
      python -c 'import accelerate, torch, transformers; print(\"torch\", torch.__version__, \"cuda_available\", torch.cuda.is_available()); print(\"transformers\", transformers.__version__); print(\"accelerate\", accelerate.__version__); assert torch.cuda.is_available(), \"The preliminary filler gate requires a CUDA GPU.\"' && \
      nvidia-smi && \
      python preliminary_filler_gate/run.py validate --config '$CONFIG_PATH' && \
      python preliminary_filler_gate/run.py run \
        --config '$CONFIG_PATH' \
        --output-dir '$RUN_DIR'"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
