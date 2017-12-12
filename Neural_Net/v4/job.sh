#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00:00
#SBATCH --job-name=binaural
#SBATCH --output=binaural_200000_%j.txt
#SBATCH -e error_200000_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py