#!/bin/bash

#----------------------------------------------------
# Slurm job script for TACC Frontera CLX nodes
#
#   *** Run EGNO Mocap with torch.compile ***
# 
# Notes:
#   -- Launch this script by executing
#      "sbatch compilers-sp25/egno-mocap-compiled.sh" from your
#       $WORK directory
#
#----------------------------------------------------

#SBATCH -J egnomocap       # Job name
#SBATCH -o slurm-out/egno-mocap-out/compiled/egnomocap.o%j   # Name of stdout output file
#SBATCH -e slurm-out/egno-mocap-out/compiled/egnomocap.e%j   # Name of stderr error file
#SBATCH -p rtx             # Queue (partition) name
#SBATCH -N 4               # Total # of nodes 
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=andrewnguyen@utexas.edu

# Set the TORCH_LOGS environment variable
export TORCH_LOGS="+dynamo,guards"

# Any other commands must follow all #SBATCH directives...
source "${WORK}/compilers-sp25/venv-1/bin/activate"

srun python3 -u ${WORK}/EGNO/compiled_mocap.py --config_by_file ${WORK}/compilers-sp25/config_mocap_no.json --outf ${WORK}/slurm-out/egno-mocap-out/compiled