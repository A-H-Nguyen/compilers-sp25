#!/bin/bash

#----------------------------------------------------
# Slurm job script for TACC Frontera CLX nodes
#
#   *** Run EGNO Mocap with MPI ***
# 
# Notes:
#   -- Launch this script by executing
#      "sbatch egno-mocap.sh" on a Frontera login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per CLX node: 56
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J egnomocap       # Job name
#SBATCH -o slurm-out/egnomocap.o%j   # Name of stdout output file
#SBATCH -e slurm-out/egnomocap.e%j   # Name of stderr error file
#SBATCH -p rtx             # Queue (partition) name
#SBATCH -N 4               # Total # of nodes 
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=andrewnguyen@utexas.edu

# Any other commands must follow all #SBATCH directives...
source "${WORK}/compilers-sp25/venv-1/bin/activate"

# this will need to be changed
srun python -u main_mocap_no.py --config_by_file --outf $log_dir
