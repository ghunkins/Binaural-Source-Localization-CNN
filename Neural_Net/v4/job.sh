#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-04:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=10db
#SBATCH --output=10db_%j.txt
#SBATCH -e 10db_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py