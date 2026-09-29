#!/bin/bash

# Run lightweight DAG steps on the submit host so they do not wait for an
# execute slot. Any failure is returned to DAGMan and blocks the next GPU stage.
set -euo pipefail

CONDOR_USER="${1:?cluster username is required}"
ACTION="${2:?local action is required}"
REPO_ROOT="/nethome/${CONDOR_USER}/Faithfulness-Safety"
VENV="/nethome/${CONDOR_USER}/.venv"
EXPERIMENT_DIR="experiments/goldreich_parity/qwen_attempts/reported_diagnostics/final_composition"
DATA_ROOT="generated_data/parity_goldreich_final/seed_0"
ARTIFACTS_ROOT="artifacts/ce_parity_goldreich_final"

cd "$REPO_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python environment: $VENV" >&2
  exit 2
fi
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

echo "Host: $HOSTNAME"
echo "Local DAG action: $ACTION"

case "$ACTION" in
  prepare)
    python "$EXPERIMENT_DIR/generate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --output-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/validate_data.py" --config "$EXPERIMENT_DIR/config.json" \
      --data-root "$DATA_ROOT"
    python "$EXPERIMENT_DIR/check_gate.py" --kind mechanism \
      --output "$ARTIFACTS_ROOT/mechanism_gate.json"
    ;;
  positions_gate)
    python "$EXPERIMENT_DIR/check_gate.py" --kind positions \
      --output "$ARTIFACTS_ROOT/positions_gate.json"
    ;;
  summarize)
    python "$EXPERIMENT_DIR/summarize.py" --config "$EXPERIMENT_DIR/config.json" \
      --artifacts-root "$ARTIFACTS_ROOT"
    ;;
  *)
    echo "Unknown local action: $ACTION" >&2
    exit 2
    ;;
esac
