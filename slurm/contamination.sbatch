#!/bin/bash
# Contamination test (a), one array task per model.
# Submit from the repo root:  sbatch slurm/contamination.sbatch
#SBATCH --job-name=wm-contam
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
MODELS=(qwen-base qwen-instruct llama-base llama-instruct mistral-base mistral-instruct)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# CHANGE_ME: activate your environment (module load / conda activate / source .venv/bin/activate)
# HF_TOKEN and HF_HOME come from .env (see .env.example); HF_HOME falls back to scratch.
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN not set (put it in .env)}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

python contamination.py --model "$MODEL"
