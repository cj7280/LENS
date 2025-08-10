#!/bin/bash

#SBATCH --job-name=render_beads
#SBATCH --account=pnlp
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
# SBATCH --constraint=gpu80

#SBATCH --output=slurm/slurmarray_%A/slurm_%A_%a.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=cj7280@princeton.edu

source /u/cj7280/.bashrc
conda activate epr

mkdir -p slurm/slurmarray_${SLURM_ARRAY_JOB_ID}

python dataset_rendered.py --config_file config.yml --job_index $SLURM_ARRAY_TASK_ID --slurm_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID
