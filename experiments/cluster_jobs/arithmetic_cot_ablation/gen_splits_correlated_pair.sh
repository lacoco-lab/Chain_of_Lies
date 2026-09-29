#!/bin/bash
# Generate large unique train/val splits for one seed.

set -e

CONDOR_USER="${1:-$USER}"
SEED="${2:-0}"
source /nethome/$CONDOR_USER/.venv/bin/activate

echo "Host: $HOSTNAME"
echo "Generating active piggyback/control prompt splits for seed=$SEED..."

python scripts/shared/data/generate_prompt_splits.py   --variant all_active   --train-n 10000   --val-n 1000   --seed "$SEED"   --output-root "generated_data/prompt_splits/seed_$SEED"
