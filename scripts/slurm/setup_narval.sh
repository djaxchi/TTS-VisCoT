#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Run this ON THE NARVAL LOGIN NODE before submitting any jobs.
# Login nodes have internet; compute nodes do not.
#
# Usage:
#   ssh dchikhi@narval.alliancecan.ca
#   cd ~/code/TTS-VisCoT
#   bash scripts/slurm/setup_narval.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=== Narval setup for TTS scale experiment ==="
echo ""

# ── 1. Environment ──────────────────────────────────────────────────────────
echo "[1/5] Loading modules and activating virtualenv..."
module load python/3.11 cuda/12.2
source $HOME/envs/viscot/bin/activate

export HF_HOME=$SCRATCH/hf_cache
mkdir -p $HF_HOME

# ── 2. Install/update the repo ──────────────────────────────────────────────
echo "[2/5] Installing repo in editable mode..."
pip install -e . --quiet

# ── 3. Download model weights to $SCRATCH/hf_cache ─────────────────────────
echo "[3/5] Downloading model weights (this may take a while on first run)..."
python -c "
from huggingface_hub import snapshot_download
print('  Downloading Qwen2.5-VL-3B-Instruct...')
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct')
print('  Downloading GRIT-20-Qwen2.5-VL-3B...')
snapshot_download('yfan1997/GRIT-20-Qwen2.5-VL-3B')
print('  Done.')
"

# ── 4. Pre-download hard_bench images ───────────────────────────────────────
echo "[4/5] Pre-downloading hard_bench images from HuggingFace..."
python -c "
from src.data.datasets.viscot_benchmark import load_task
for task in ['vqa', 'ocr', 'counting']:
    examples = load_task(task)
    print(f'  {task}: {len(examples)} images cached')
print('  Done. Images saved to data/hard_bench/images/')
"

# ── 5. Create output directories and logs ───────────────────────────────────
echo "[5/5] Creating output directories..."
mkdir -p $SCRATCH/results/tts_scale
mkdir -p $SCRATCH/results/tts_scale_t0
mkdir -p logs

# Symlink results into repo so the script finds them
ln -sfn $SCRATCH/results/tts_scale results/tts_scale
ln -sfn $SCRATCH/results/tts_scale_t0 results/tts_scale_t0

echo ""
echo "=== Setup complete ==="
echo ""
echo "To submit all 4 jobs:"
echo "  sbatch scripts/slurm/tts_scale_qwen_standard.sh"
echo "  sbatch scripts/slurm/tts_scale_qwen_t0.sh"
echo "  sbatch scripts/slurm/tts_scale_grit_standard.sh"
echo "  sbatch scripts/slurm/tts_scale_grit_t0.sh"
echo ""
echo "To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/tts_grit_std_*.out"
echo ""
echo "To download results after completion:"
echo "  # From your LOCAL machine:"
echo "  scp -r dchikhi@narval.alliancecan.ca:\$SCRATCH/results/tts_scale* results/"
