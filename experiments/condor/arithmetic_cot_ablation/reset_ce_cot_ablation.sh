#!/bin/bash
# Clear generated_data/results for the CE CoT ablation experiment.

set -e

CONDOR_USER="${1:-$USER}"
source /nethome/$CONDOR_USER/.venv/bin/activate

python scripts/reset_ce_cot_ablation.py
