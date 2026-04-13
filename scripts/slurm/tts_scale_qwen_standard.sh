#!/bin/bash
#SBATCH --account=def-azouaq_gpu
#SBATCH --job-name=tts_qwen_std
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dchikhi@polymtl.ca

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.2
source $HOME/envs/viscot/bin/activate

# Offline mode — compute nodes have no internet
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=$HOME/code/TTS-VisCoT

# Results go to $SCRATCH (large quota), symlinked into repo
RESULTS_DIR=$SCRATCH/results/tts_scale
mkdir -p $RESULTS_DIR
mkdir -p $REPO/logs

# Symlink so the script writes to $SCRATCH transparently
ln -sfn $RESULTS_DIR $REPO/results/tts_scale

cd $REPO

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting Qwen3B standard recipe at $(date)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

python experiments/run_tts_scale.py \
    --model qwen3b \
    --recipe standard

echo "Finished at $(date)"
