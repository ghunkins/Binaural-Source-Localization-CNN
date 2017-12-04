#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem-per-cpu=10gb
#SBATCH -t 0-04:00:00
#SBATCH --job-name=binaural
#SBATCH --output=binaural.txt
#SBATCH -e error.txt
#SBATCH --mail-type=begin
#SBATCH --mail-user=ghunkins@u.rochester.edu
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

source activate keras
python neuralnet.py