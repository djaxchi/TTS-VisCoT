#!/usr/bin/env bash
# run_ablations.sh — run baseline and TTS experiments back-to-back
set -euo pipefail

MAX_SAMPLES="${MAX_SAMPLES:-50}"

echo "=== Baseline ==="
python experiments/run_viscot.py --config configs/experiments/baseline.yaml \
    dataset.max_samples="$MAX_SAMPLES"

echo ""
echo "=== TTS Scaling ==="
python experiments/run_viscot.py --config configs/experiments/tts.yaml \
    dataset.max_samples="$MAX_SAMPLES" method=tts_scaling

echo ""
echo "✓ Ablations complete. Results are in results/."
