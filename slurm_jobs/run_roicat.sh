#!/bin/bash

#SBATCH --job-name=submitit
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.kuroda@ucl.ac.uk

source /etc/profile.d/modules.sh
echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

export PYTHONUNBUFFERED=1

module load miniconda
conda activate photon-mosaic-dev

srun python ./run_roicat.py