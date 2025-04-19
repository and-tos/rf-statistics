#!/bin/bash

#BSUB -J pdb_quality
#BSUB -n 32
#BSUB -W 96:00
#BSUB -q long
#BSUB -R "rusage[mem=32G/host]"
#BSUB -o lsf_%J_pdb_quality.out

conda activate rf-statistics3.3

python rf-statistics/rf_statistics/database_utils/pdb_quality.py --nproc 32
