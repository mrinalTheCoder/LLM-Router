#!/bin/bash
#SBATCH -n8
#SBATCH --mem-per-cpu=2G
#SBATCH -t8:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH -o/home/hice1/mjain330/scratch/logs/report-%j.out

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate llm-router

wandb agent llm-router/llm-router/z2wwjnci
