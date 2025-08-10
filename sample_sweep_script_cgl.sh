#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --account=pnlp

# Specify GPU mem.
# All experiments in paper can fit on a single A6000 GPU,
# but not generally true (see fits_in_vram flag in ymls and utils.py parser).
#SBATCH --gres=gpu:a6000:1

#SBATCH --output=slurm/slurmarray_%A/slurm_%a.out
# SBATCH --constraint=gpu80
# SBATCH --nodelist=node211
#SBATCH --array=1-2
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=USER_EMAIL@SCHOOL.EDU

# Activate conda environment.
source /u/cj7280/.bashrc
conda activate env_jrl

# Run the sweep.
python sweep.py --config_file "ymls/CGL_figs_(m4)_config_training.yml" --job_index ${SLURM_ARRAY_TASK_ID} --slurm_id ${SLURM_ARRAY_JOB_ID}