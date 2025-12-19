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

# Configuration file passed as environment variable (default to TP=4)
CONFIG=${CONFIG:-"E1.3_llama8b_tp4.yaml"}
EXP_NAME=$(basename $CONFIG .yaml)

export NCCL_DEBUG=INFO

# Ensure trace directory exists
mkdir -p trace_collection/${EXP_NAME}

# Run with Nsys profiling (capture full run; no CUDA capture-range gating)
srun nsys profile \
    --force-overwrite=true \
    -o trace_collection/${EXP_NAME}/nsys_report \
    --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-device=all \
    python vllm_profiler.py --config experiments/configs/${CONFIG}

# Export CUDA GPU trace to CSV for downstream tools
NSYS_REP=$(ls trace_collection/${EXP_NAME}/nsys_report.*rep | head -n1)
if [ -z "$NSYS_REP" ]; then
    echo "ERROR: Nsys report not found under trace_collection/${EXP_NAME}"
    exit 1
fi

nsys stats \
    --force-overwrite=true \
    --report cuda_gpu_trace \
    --format csv \
    -o trace_collection/${EXP_NAME}/cuda_gpu_trace \
    "$NSYS_REP"

echo "Nsys profiling for ${EXP_NAME} completed!"
