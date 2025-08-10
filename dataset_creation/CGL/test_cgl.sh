#!/bin/bash
#SBATCH --job-name=CGL
#SBATCH --account=pnlp
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-29 # 0 indexed here
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G

#SBATCH --output=slurm/slurmarray_%A/slurm_%A_%a.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=cj7280@princeton.edu

# Make dir of slurm array A
mkdir -p slurm/slurmarray_${SLURM_ARRAY_JOB_ID}

source /u/cj7280/.bashrc
conda activate env_jrl

# Run the sweep 
python dataset_test.py --config_file params.yml --job_index ${SLURM_ARRAY_TASK_ID} --slurm_id ${SLURM_ARRAY_JOB_ID}