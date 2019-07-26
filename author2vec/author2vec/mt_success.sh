#!/bin/bash

### To run multiple jobs
#SBATCH --array=0-2

### Set the job name
#SBATCH -J authorMT

### To send email for the job
#SBATCH --mail-user=smaharjan2@uh.edu
###SBATCH --mail-type=begin    # email me when the job starts
###SBATCH --mail-type=end      # email me when the job finishes

### logdir
#SBATCH -o /uhpc/solorio/suraj/authors/logs/normalized/authormt_%A_%a.log


### Specify the number of cpus for your job.
#SBATCH -N 1             # total number of nodes
#SBATCH --ntasks-per-node 1  # number of processors per node
#SBATCH --mem 64000           # memory you expect to use in mb.
#SBATCH -t 60:00:00         # anticipated run-time for your job, (HH:MM:SS)


echo ID:$SLURM_ARRAY_TASK_ID

source activate py35

python  manage.py run_success_multitask --f   $SLURM_ARRAY_TASK_ID
