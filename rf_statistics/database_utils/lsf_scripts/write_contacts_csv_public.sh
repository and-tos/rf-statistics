#!/bin/bash

#BSUB -J assign_rf
#BSUB -n 32
#BSUB -R "rusage[mem=16G/host]"
#BSUB -W 96:00
#BSUB -q long
#BSUB -o lsf_%J_assign.out

conda activate rf-statistics3.3

export_contacts_surfaces --nproc $LSB_DJOB_NUMPROC -db public
