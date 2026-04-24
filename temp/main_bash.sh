#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH -c 1
#SBATCH --time=72:00:00
#SBATCH --output=/home/ikvasilev/PaTHoP1/assets/logs/prediction_multiple_ts_%j.log
module purge
module load Python/Miniconda_v25
source activate nk_env_clean

scontrol update job=$SLURM_JOB_ID JobName="prediction"
#
python ./main.py