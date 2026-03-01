#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abybordi@gradcenter.cuny.edu

module load anaconda3

source activate py371



python predict_marc.py og_teretoxins_1.csv > pred_marc_sav_out.txt 2>&1
