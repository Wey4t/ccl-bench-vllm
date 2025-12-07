#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J E1.1_llama8b_baseline
#SBATCH -o logs/E1.1_%j.out
#SBATCH -e logs/E1.1_%j.err

# E1.1: Llama-8B | TP=1 | 1 GPU - Baseline

module load python
module load cuda/12.4

# Activate environment
source .venv/bin/activate

# Create log directory
mkdir -p logs

# Set NCCL environment variables for profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export HF_HOME=/pscratch/sd/n/nw388/huggingface
export TORCH_EXTENSIONS_DIR=$SCRATCH/torch_extensions
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# Run profiling
srun python vllm_profiler.py \
    --config experiments/configs/E1.1_llama-3.1-8b.yaml

echo "Experiment E1.1 completed!"
