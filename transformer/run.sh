#!/bin/bash -l
#SBATCH --job-name=df
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=icsnode06

conda activate venv
python3 /home/zang/language_model/01_transformer/train.py
