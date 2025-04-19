#!/bin/bash
#BSUB -J proasis_to_csd
#BSUB -n 2
#BSUB -R "rusage[mem=8G/host]"
#BSUB -q long
#BSUB -W 240:00
#BSUB -o lsf_%J_proasis_to_csd.out

conda activate rf-statistics3.3

python rf-statistics/rf_statistics/database_utils/ccdc_library.py -i public
