#!/bin/bash

# Job name
#SBATCH -J "train_YOLO"

# Batch output file
#SBATCH --output run_torch.o%J

# Batch error file
#SBATCH --error run_torch.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition 2tcourt

# GPUs per compute node
#   8 (maximum) for gpu
#   8 (maximum) for hpda

# ----------------------------
# processes / tasks
#SBATCH -n 1

# ----------------------------
# CPUs per task
# Set the number of cpu in proportion to the number of GPU's devices :
#   gpu: until 8 cores / device
#   hpda: until 8 cores / device
#SBATCH --cpus-per-task 16

# ------------------------
# Job time (hh:mm:ss)
#SBATCH --time 12:00:00
# ------------------------

##SBATCH --mail-type ALL
# User e-mail address
##SBATCH --mail-user tom.poget@insa-rouen.fr

module purge
module load aidl/pytorch/2.6.0-py3.12-cuda12.6

rm -rf .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
srun uv run finetune.py
