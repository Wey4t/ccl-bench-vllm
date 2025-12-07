salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 1 --account m4999
module load python
module load cuda/12.4
conda activate /pscratch/sd/n/nw388/vllm-profiling
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL 
export HF_HOME=/pscratch/sd/n/nw388/huggingface
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_MOE_FP16=1

nsys profile\
    -o trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/nsys_%h_%p \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --stats=true \
    python vllm_profiler.py --config experiments/configs/E3.3_deepseek-v2-lite.yaml

python experiments/generate_workload_card.py \
    --config experiments/configs/E3.3_deepseek-v2-lite.yaml \
    --output-dir trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3


nw_ep_analysis/plot_moe_stats.py trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3 --output-dir nw_ep_analysis/plots_E3.3