#!/bin/bash
set -euo pipefail

CONDOR_USER="${1:?cluster username required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"

cd "$REPO_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python environment: $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python experiments/legacy/scaling_regime/calibration/hard_checkpoint_length_sweep/generate_splits.py
python experiments/legacy/scaling_regime/calibration/hard_checkpoint_length_sweep/validate.py

echo "Hard-checkpoint length-sweep generation and validation complete."
