#!/bin/bash
#SBATCH --job-name=test_beads
#SBATCH --account=pnlp
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a6000:1
# BATCH --nodes=1
#SBATCH --array=0-9 # 0 indexed here
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

#SBATCH --output=slurm/slurmarray_%A/slurm_%A_%a.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=cj7280@princeton.edu

# Make dir of slurm array A
mkdir -p slurm/slurmarray_${SLURM_ARRAY_JOB_ID}

source /u/cj7280/.bashrc
conda activate epr

export JAX_ENABLE_X64=True

# Run the sweep 
python dataset_test.py --config_file params_N=32,64.yml --job_index ${SLURM_ARRAY_TASK_ID} --slurm_id ${SLURM_ARRAY_JOB_ID}