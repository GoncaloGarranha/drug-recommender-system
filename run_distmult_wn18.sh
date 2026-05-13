#!/bin/bash

set -e

CONFIG="models/rgcn_schlichtkrull/configs/wn18.yaml"
SEED=42
EPOCHS=50  # Debug: fewer epochs for quick testing

echo "======================================"
echo "DistMult Standalone - WN18"
echo "======================================"

# 1. Update config for DistMult training
echo "Updating config..."
python -c "
import yaml
with open('$CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['training']['max_epochs'] = $EPOCHS
cfg['training']['epochs'] = $EPOCHS
cfg['training']['seed'] = $SEED
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print(f'Config updated: epochs={$EPOCHS}, seed={$SEED}')
"

# 2. Train DistMult
echo ""
echo "Training DistMult on WN18..."
TRAIN_OUTPUT=$(python -m models.rgcn_schlichtkrull.src.train_distmult \
  --config "$CONFIG" \
  --eval_mode filtered)

echo "$TRAIN_OUTPUT"

# Extract checkpoint path from run output (last created run directory)
RUN_DIR=$(ls -dt models/rgcn_schlichtkrull/runs/*_distmult | head -1)
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo ""
echo "Checkpoint: $CHECKPOINT"

# 3. Evaluate and export to CSV
echo ""
echo "Evaluating and exporting results..."
python -m models.rgcn_schlichtkrull.src.evaluate_distmult \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --dataset wn18 \
  --seed $SEED \
  --output_csv benchmarks_schlichtkrull.csv

echo ""
echo "======================================"
echo "✓ Complete! Results in benchmarks_schlichtkrull.csv"
echo "======================================"
