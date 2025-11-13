#!/bin/bash
#SBATCH --job-name=mnist_entanglement_analysis
#SBATCH --output=/nfs/home/lverho/classwise_tda/scripts/logs/mnist_entanglement_analysis.out
#SBATCH --error=/nfs/home/lverho/classwise_tda/scripts/logs/mnist_entanglement_analysis.err
#SBATCH --partition=day-long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
echo "Starting process"
conda activate classwise_tda
python /nfs/home/lverho/classwise_tda/scripts/mnist_entanglement_analysis.py
