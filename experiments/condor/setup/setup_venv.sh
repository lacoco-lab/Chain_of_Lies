#!/bin/bash
# One-time setup: create venv and pip install on a worker with enough memory.
# Uses system python3 (no "module" required). Submit with: condor_submit experiments/condor/setup/setup_venv.sub

echo "=== Setup job started at $(date) on $(hostname) ==="

set -e
# Condor may not set USER; pass username as first arg from setup_venv.sub
CONDOR_USER="${1:-$USER}"
echo "Python3: $(which python3); $(python3 --version)"

VENV=/nethome/$CONDOR_USER/.venv
if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV ..."
  python3 -m venv "$VENV"
else
  echo "Using existing venv at $VENV"
fi
source "$VENV/bin/activate"

echo "Installing requirements (this may take 10–20 min) ..."
pip install -r /nethome/$CONDOR_USER/Faithfulness-Safety/requirements.txt
echo "=== Done. Venv ready at $VENV at $(date) ==="
