#!/bin/bash
#SBATCH --job-name=binaural
#SBATCH --output=binaural.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=50GB
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py