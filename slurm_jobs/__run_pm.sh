source /etc/profile.d/modules.sh
echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

module load miniconda
conda activate photon-mosaic

PYTHONUNBUFFERED=1 photon-mosaic --jobs 10 --rerun-incomplete