#!/bin/bash
#BSUB -J proasis_to_rdkit
#BSUB -n 24
#BSUB -R "rusage[mem=64G/host]"
#BSUB -W 20:00
#BSUB -o lsf_%J_proasis_to_rdkit.out

conda activate rf-statistics3.3

python rf-statistics/rf_statistics/database_utils/rdkit_library.py -i internal -n $LSB_DJOB_NUMPROC
