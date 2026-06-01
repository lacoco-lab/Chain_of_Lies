#!/bin/bash
# Summarize CE CoT ablation results.

set -e

CONDOR_USER="${1:-$USER}"
source /nethome/$CONDOR_USER/.venv/bin/activate

python scripts/summarize_ce_cot_ablation.py \
  --root artifacts/ce_cot_ablation
