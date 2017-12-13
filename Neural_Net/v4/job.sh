#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=log_binaural
#SBATCH --output=log_binaural_traintest_sep_%j.txt
#SBATCH -e log_error_traintest_sep_%j.txt
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py