#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 10                    
#SBATCH --mem=125gb            
#SBATCH --ntasks-per-node=2
#SBATCH -t 0-04:00:00
#SBATCH --job-name=final
#SBATCH --output=final.txt
#SBATCH -e error.txt
#SBATCH --mail-type=begin
#SBATCH --mail-user=ghunkins@u.rochester.edu
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:2

source activate keras
python neuralnet.py