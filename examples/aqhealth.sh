#!/bin/bash
#SBATCH --mem=90G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 24 
#SBATCH -t 01-00:00:00
#SBATCH -p normal
#SBATCH -J Glob

source /opt/conda/bin/activate mort_env

python3 -u Mortalities.py


