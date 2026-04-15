#!/bin/bash
#SBATCH -n4
#SBATCH --mem-per-cpu=1G
#SBATCH -t1:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH -o~/scratch/logs/report-%j.out

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load ollama
conda activate llm-router

srun run_single_llm_outouts.sh --model "llama3-chatqa:8b"
srun run_single_llm_outouts.sh --model "llama3.2:3b"
srun run_single_llm_outouts.sh --model "qwen2.5:3b"
