#!/bin/bash

set -euo pipefail

LENGTH="${1:?Usage: run_generate_scale.sh LENGTH}"
JOB_ROOT="${_CONDOR_SCRATCH_DIR:-$PWD}"
EXPERIMENT="experiments/goldreich_parity/transformer/main_experiment"
CONFIG="$EXPERIMENT/scale_config.json"
DATA_ROOT="generated_data/parity_goldreich_toy_transformer_scale/seed_0"
LENGTH_ROOT="$DATA_ROOT/n$LENGTH"

cd "$JOB_ROOT"
export PYTHONUNBUFFERED=1

run_with_heartbeat() {
  local label="$1"
  shift
  "$@" &
  local child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep 60
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "$label is still running at $(date --iso-8601=seconds)"
      du -sh "$LENGTH_ROOT" 2>/dev/null || true
    fi
  done
  wait "$child_pid"
}

echo "Generating and validating Goldreich data for N=$LENGTH"
run_with_heartbeat "Generation for N=$LENGTH" \
  python3 "$EXPERIMENT/generate_data.py" \
  --config "$CONFIG" \
  --output-root "$DATA_ROOT" \
  --length "$LENGTH"

run_with_heartbeat "Validation for N=$LENGTH" \
  python3 "$EXPERIMENT/validate_data.py" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --length "$LENGTH"

# Keep the per-length manifest beside the split files so independently
# transferred jobs remain self-describing and auditable.
cp "$DATA_ROOT/manifest.json" "$LENGTH_ROOT/manifest.json"

test -s "$LENGTH_ROOT/train.jsonl.gz"
test -s "$LENGTH_ROOT/validation.jsonl.gz"
test -s "$LENGTH_ROOT/test.jsonl.gz"
test -s "$LENGTH_ROOT/manifest.json"

echo "Generation and validation succeeded for N=$LENGTH"
du -sh "$LENGTH_ROOT"
