#!/bin/bash
#SBATCH --job-name=binaural
#SBATCH --output=binaural.txt
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py