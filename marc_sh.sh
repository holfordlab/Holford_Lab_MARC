#!/bin/bash
#SBATCH -N 1
#SBATCH -p EM
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abybordi@gradcenter.cuny.edu

module load anaconda3

source activate py371



python marc.py > marc_sav_out.txt 2>&1
