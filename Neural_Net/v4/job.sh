#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-04:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=20db
#SBATCH --output=20db_%j.txt
#SBATCH -e 20db_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py