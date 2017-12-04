#!/bin/bash
#SBATCH --job-name=binaural
#SBATCH --output=binaural.txt
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

python neuralnet.py