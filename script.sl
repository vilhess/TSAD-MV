#!/bin/bash

## HELP : /soft/slurm/Modeles_scripts/pytorch

#SBATCH --output logs/run_%J
#SBATCH --error logs/run_%J

#SBATCH --partition gpu_debug
##SBATCH --partition gpu_h200
##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-task 16
##SBATCH --time 24:00:00

module purge
module load aidl/pytorch/2.6.0-cuda12.6
cp -r ./ $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR/

echo Working directory : $PWD
echo Submit directory : $SLURM_SUBMIT_DIR

srun python -m  benchmark_exp.Run_Detector_U_Missing \
    --AD_Name MTAD
