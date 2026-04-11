#!/bin/bash
#SBATCH --account=def-azouaq
#SBATCH --job-name=deepeyes_treebench
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.2
source $HOME/envs/viscot/bin/activate

# Model weights must be pre-downloaded on the login node:
#   export HF_HOME=$SCRATCH/hf_cache
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('honglyhly/DeepEyesV2_7B_1031')"
export HF_HOME=$SCRATCH/hf_cache

# HuggingFace datasets cache (TreeBench images downloaded on first run, then offline)
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO=$HOME/code/TTS-VisCoT
RESULTS=$SCRATCH/results/treebench

mkdir -p $RESULTS
mkdir -p $REPO/logs

cd $REPO

# ── Run ────────────────────────────────────────────────────────────────────────
python experiments/run_deepeyes_treebench.py \
    --n 10 \
    --output "$RESULTS/deepeyes_treebench_10.jsonl"
