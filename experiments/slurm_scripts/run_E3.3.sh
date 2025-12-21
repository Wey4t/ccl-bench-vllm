salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 4 --account m4999
module load python
module load cuda/12.4
conda activate /pscratch/sd/n/nw388/vllm-profiling
export HF_HOME=/pscratch/sd/n/nw388/huggingface
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL 

export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TMPDIR=/pscratch/sd/n/nw388/tmp
export VLLM_USE_V1=1
export VLLM_RINGBUFFER_WARNING_INTERVAL=600
export GLOO_TIMEOUT_SECONDS=1200
export VLLM_RPC_TIMEOUT=1200

# Run with internal vLLM profiler (Torch Profiler / Kineto)
python vllm_profiler.py --config experiments/configs/E3.3_deepseek-v2-lite.yaml

python experiments/generate_workload_card.py \
    --config experiments/configs/E3.3_deepseek-v2-lite.yaml \
    --output-dir trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3


python nw_ep_analysis/plot_moe_stats.py trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3 --output-dir nw_ep_analysis/plots_E3.3
python scripts/analyze_trace_timeline.py \
    trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3 \
    --output-dir trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/plots

# Export nsys trace to JSON for analysis
# Handle multiple nsys-rep files (one per process)
for f in trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/nsys_*.nsys-rep; do
    nsys export --type json --output "${f%.*}.json" "$f" --force
done

python experiments/generate_workload_card.py \
    --config experiments/configs/E3.3_deepseek-v2-lite.yaml \
    --output-dir trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3


python nw_ep_analysis/plot_moe_stats.py trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3 --output-dir nw_ep_analysis/plots_E3.3
python scripts/analyze_trace_timeline.py \
    trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3 \
    --output-dir trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/plots



nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --output=trace_collection/nsys-4/ep4-batch2048 \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  python3 benchmark_simple.py experiments/configs/E3.3_deepseek-v2-lite.yaml


nsys export -t sqlite -o /pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4-batch256.sqlite /pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4-batch256.nsys-rep
python experiments/analyze_trace.py