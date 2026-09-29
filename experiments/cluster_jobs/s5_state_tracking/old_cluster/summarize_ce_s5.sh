#!/bin/bash
# Summarize CE S5 results.

set -e

CONDOR_USER="${1:-$USER}"
ARTIFACTS_ROOT="${2:-artifacts/ce_s5}"
source /nethome/$CONDOR_USER/.venv/bin/activate

python scripts/shared/evaluation/summarize_ce.py \
  --root "$ARTIFACTS_ROOT" \
  --modes answer_only,public_cot \
  --variants s5_piggyback,s5_control \
  --piggyback-variant s5_piggyback \
  --control-variant s5_control

