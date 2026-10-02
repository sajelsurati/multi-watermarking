#!/bin/bash
# Main experiment: one array task per (model, language) = 6 x 10 = 60 tasks.
# Each task runs all 7 conditions x 2 stop modes and skips finished outputs,
# so resubmitting the same array resumes where it stopped.
# Submit from the repo root:  sbatch slurm/generate.sbatch
# Single pair:                sbatch --array=<idx> slurm/generate.sbatch   (idx = model_idx*10 + lang_idx)
#SBATCH --job-name=wm-gen
#SBATCH --array=0-59%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
MODELS=(qwen-base qwen-instruct llama-base llama-instruct mistral-base mistral-instruct)
LANGS=(en es ru zh ja ar hi tr sw yo)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 10))]}
LANG_=${LANGS[$((SLURM_ARRAY_TASK_ID % 10))]}

# CHANGE_ME: activate your environment (module load / conda activate / source .venv/bin/activate)
# HF_TOKEN and HF_HOME come from .env (see .env.example); HF_HOME falls back to scratch.
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN not set (put it in .env)}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}

python generate.py --model "$MODEL" --lang "$LANG_"
