#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --qos=extuser1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --error=minilm.%J.err
#SBATCH --output=minilm.%J.out
#SBATCH --partition=nltmp
WANDB_MODE=offline python3 train.py
