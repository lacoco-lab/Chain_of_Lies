#!/bin/bash
# Summarize CE S5 results.

set -e

CONDOR_USER="${1:-$USER}"
ARTIFACTS_ROOT="${2:-artifacts/ce_s5}"
source /nethome/$CONDOR_USER/.venv/bin/activate

python scripts/summarize_ce_results.py \
  --root "$ARTIFACTS_ROOT" \
  --modes answer_only,public_cot \
  --variants s5_piggyback,s5_control \
  --piggyback-variant s5_piggyback \
  --control-variant s5_control

