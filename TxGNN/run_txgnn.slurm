#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonstar6 SKX nodes
#
#   *** Serial Job on SKX Normal Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Lonstar6 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J TxGNN                        # Job name
#SBATCH -o TxGNN.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e TxGNN.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p rtx                   # Queue (partition) name
#SBATCH -N 2                            # Total # of nodes (must be 1 for serial)
#SBATCH -n 2                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=mp46753@utexas.edu
#SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)


# Other commands must follow all #SBATCH directives...

#module load intel # Comment out for now, you might check available modules using "module avail"

# Launch serial code...
#cd $WORK/Lab3/benchmarks/401.bzip2
#$WORK/Lab3/pin-3.30/pin -t $WORK/Lab3/pin-3.30/source/tools/ManualExamples/obj-intel64/inscount0.so -- ./bzip2 input.combined 200
cd ${WORK}/ML_Project
source ./miniconda3/bin/activate
conda activate txgnn_env

cd ./TxGNN
#(TORCH_LOGS="recompiles" srun python3 test.py --outf ${WORK}/slurm-out/TxGNN-out/run) &> out.log
(TORCH_LOGS="recompiles" srun python3 test_txgnn.py) &> out_txgnn.log


# ---------------------------------------------------
