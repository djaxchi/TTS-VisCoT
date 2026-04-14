#!/bin/bash
#SBATCH --account=def-azouaq_gpu
#SBATCH --job-name=rerank_grit_std
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dchikhi@polymtl.ca

module load python/3.11 cuda/12.2 gcc arrow
source $HOME/envs/viscot/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO=$HOME/code/TTS-VisCoT

# Link scale results so the script can find them
ln -sfn $SCRATCH/results/tts_scale    $REPO/results/tts_scale
ln -sfn $SCRATCH/results/tts_scale_t0 $REPO/results/tts_scale_t0

RERANK_DIR=$SCRATCH/results/rerank
mkdir -p $RERANK_DIR $REPO/logs
ln -sfn $RERANK_DIR $REPO/results/rerank

cd $REPO

echo "Starting rerank — GRIT standard at $(date)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

python experiments/run_rerank.py \
    --source results/tts_scale/grit_results.jsonl

echo "Finished at $(date)"
