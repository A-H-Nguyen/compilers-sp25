#!/bin/bash

#SBATCH -J torchtest       # Job name
#SBATCH -o slurm-out/torchtest.o%j   # Name of stdout output file
#SBATCH -e slurm-out/torchtest.e%j   # Name of stderr error file
#SBATCH -p rtx             # Queue (partition) name
#SBATCH -N 4               # Total # of nodes 
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=andrewnguyen@utexas.edu

# Any other commands must follow all #SBATCH directives...
source "${WORK}/compilers-sp25/venv-1/bin/activate"

srun torchrun --nproc_per_node=4 ${WORK}/examples/distributed/ddp-tutorial-series/multigpu_torchrun.py 50 10
