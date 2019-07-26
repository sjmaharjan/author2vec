#!/bin/bash

### To run multiple jobs
#SBATCH --array=0-44

### Set the job name
#SBATCH -J AA

### To send email for the job
#SBATCH --mail-user=smaharjan2@uh.edu
###SBATCH --mail-type=begin    # email me when the job starts
###SBATCH --mail-type=end      # email me when the job finishes

### logdir
#SBATCH -o /uhpc/solorio/suraj/authors/logs/aa_%A_%a.log


### Specify the number of cpus for your job.
#SBATCH -N 1             # total number of nodes
#SBATCH --ntasks-per-node 1  # number of processors per node
#SBATCH --mem 30000           # memory you expect to use in mb.
#SBATCH -t 2:00:00         # anticipated run-time for your job, (HH:MM:SS)


echo ID:$SLURM_ARRAY_TASK_ID

source activate py35

python  /uhpc/solorio/suraj/authors/author_style/code/aa_main.py   $SLURM_ARRAY_TASK_ID
