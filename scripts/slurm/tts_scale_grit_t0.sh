#!/bin/bash
#SBATCH --account=def-azouaq_gpu
#SBATCH --job-name=tts_grit_t0
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dchikhi@polymtl.ca

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.2
source $HOME/envs/viscot/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=$HOME/code/TTS-VisCoT

RESULTS_DIR=$SCRATCH/results/tts_scale_t0
mkdir -p $RESULTS_DIR
mkdir -p $REPO/logs

ln -sfn $RESULTS_DIR $REPO/results/tts_scale_t0

cd $REPO

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting GRIT T=0 ablation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

python experiments/run_tts_scale.py \
    --model grit \
    --recipe t0

echo "Finished at $(date)"
