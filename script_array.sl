#!/bin/bash

#SBATCH --job-name=ad_benchmark
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --array=0-1

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

module purge
module load aidl/pytorch/2.6.0-cuda12.6

cp -r . $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR

MODELS=(
    CRIB
    IMAD
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL"
echo "Working directory: $PWD"
echo "Submit directory: $SLURM_SUBMIT_DIR"

srun python -m benchmark_exp.Run_Detector_U_Missing \
    --AD_Name "$MODEL"
