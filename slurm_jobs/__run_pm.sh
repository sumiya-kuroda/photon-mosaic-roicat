source /etc/profile.d/modules.sh
echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

module load miniconda
conda activate photon-mosaic-dev

photon-mosaic --jobs 10 --rerun-incomplete