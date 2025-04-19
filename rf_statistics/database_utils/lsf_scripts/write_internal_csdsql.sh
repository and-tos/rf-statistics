#!/bin/bash
#BSUB -J proasis_to_csd
#BSUB -n 2
#BSUB -W 48:00
#BSUB -R "rusage[mem=16G/host]"
#BSUB -o lsf_%J_proasis_to_csd.out

conda activate rf-statistics3.3

python rf-statistics/rf_statistics/database_utils/ccdc_library.py -i internal
