#!/bin/bash
# Generate large unique S5 train/val splits for one seed.

set -e

CONDOR_USER="${1:-$USER}"
SEED="${2:-0}"
source /nethome/$CONDOR_USER/.venv/bin/activate

echo "Host: $HOSTNAME"
echo "Generating S5 piggyback/control prompt splits for seed=$SEED..."

python scripts/generate_prompt_splits.py \
  --variant all_s5 \
  --train-n 10000 \
  --val-n 1000 \
  --seed "$SEED" \
  --output-root "generated_data/prompt_splits/seed_$SEED"

python scripts/sanity_check_s5.py --split-root "generated_data/prompt_splits/seed_$SEED" --variant s5_piggyback
python scripts/sanity_check_s5.py --split-root "generated_data/prompt_splits/seed_$SEED" --variant s5_control

