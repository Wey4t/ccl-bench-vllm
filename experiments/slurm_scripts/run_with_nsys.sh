#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J nsys_profiling
#SBATCH -o logs/nsys_%j.out
#SBATCH -e logs/nsys_%j.err

# Run experiment with Nsys profiling
# Usage: sbatch --export=CONFIG=E1.3_llama8b_tp4.yaml run_with_nsys.sh

module load python
module load cuda/12.4
module load nsight-systems

source activate vllm-profiling

mkdir -p logs

# Configuration file passed as environment variable
CONFIG=${CONFIG:-"E1.1_llama8b_baseline.yaml"}
EXP_NAME=$(basename $CONFIG .yaml)

export NCCL_DEBUG=INFO

# Run with Nsys profiling
srun nsys profile \
    -o trace_collection/${EXP_NAME}/nsys_%h_%p \
    --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
    --gpu-metrics-device=all \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python vllm_profiler.py --config experiments/configs/${CONFIG}

echo "Nsys profiling for ${EXP_NAME} completed!"
