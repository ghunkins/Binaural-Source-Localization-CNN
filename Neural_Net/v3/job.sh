#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH --mem 10GB
#SBATCH --job-name=binaural
#SBATCH --output=binaural_full_%j.txt
#SBATCH -e error_full_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py