#!/bin/bash
#SBATCH -n4
#SBATCH --mem-per-cpu=4G
#SBATCH -t4:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH -o/home/hice1/mjain330/scratch/logs/router-train-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda3
conda activate llm-router

CACHED_OUTPUT_DIR="${CACHED_OUTPUT_DIR:-${HOME}/scratch/model_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/scratch/router_ppo}"
WANDB_MODE="${WANDB_MODE:-disabled}"
ENCODER_TRAINING_MODE="${ENCODER_TRAINING_MODE:-frozen}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
VAL_LIMIT="${VAL_LIMIT:-128}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000}"
MODEL_POOL="${MODEL_POOL:-llama3-chatqa:8b,llama3.2:3b,qwen2.5:7b,dolphin3:8b,granite4:3b,mathstral:7b,meditron:7b,sailor2:8b}"

mkdir -p "$OUTPUT_DIR"

python scripts/train_router_ppo.py \
    --encoder-training-mode "$ENCODER_TRAINING_MODE" \
    --outcome-source cached \
    --cached-output-dir "$CACHED_OUTPUT_DIR" \
    --train-split test \
    --val-split validation \
    --train-limit "$TRAIN_LIMIT" \
    --val-limit "$VAL_LIMIT" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --model-pool "$MODEL_POOL" \
    --output-dir "$OUTPUT_DIR" \
    --wandb-mode "$WANDB_MODE" \
    "$@"
