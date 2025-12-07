salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 1 --account m4999
module load python
module load cuda/12.4
conda activate /pscratch/sd/n/nw388/vllm-profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL 
export HF_HOME=/pscratch/sd/n/nw388/huggingface
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
python vllm_profiler.py \
    --config experiments/configs/E3.1_deepseek-v2-lite.yaml


nsys profile \
    -o trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.1/nsys_%h_%p \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    python vllm_profiler.py --config experiments/configs/E3.1_deepseek-v2-lite.yaml
