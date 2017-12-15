#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=0db
#SBATCH --output=output_0db_%j.txt
#SBATCH -e error_0db_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py