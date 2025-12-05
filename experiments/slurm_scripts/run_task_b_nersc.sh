#!/bin/bash
# Script to run Task B experiments on NERSC
# Usage: 
# 1. Allocate node: salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999_g
# 2. Run this script: bash experiments/slurm_scripts/run_task_b_nersc.sh

# Exit on error
set -e

echo "=== Setting up Environment ==="
module load conda 
conda activate sysml
pip install pandas

# Set HuggingFace Cache (Modify if needed)
export HF_HOME=/pscratch/sd/${USER:0:1}/$USER/huggingface
if [ -f "$HF_HOME/token" ]; then
    export HUGGINGFACE_HUB_TOKEN="$(cat $HF_HOME/token)"
fi

# Ensure we are in the project root
cd "$(dirname "$0")/../.."
echo "Working directory: $(pwd)"

echo "=== Running Task B Workflow ==="
# This runs E2.1, E2.2, E2.3 and generates the summary
python run_task_b.py

echo "=== Done ==="
echo "Results are in experiments/results_summary.csv"
