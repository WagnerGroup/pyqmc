#!/bin/bash
##############################
#
#PBS -l walltime=00:25:00
#
#PBS -l naccesspolicy=singleuser
#PBS -N test
#
##############################
module load intel/18.0
source ~/.bashrc
conda init bash
conda activate pyscf
cd $PBS_O_WORKDIR
python3 he.py
