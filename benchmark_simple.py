import os
import sys
import argparse
import yaml

# Explicitly unset VLLM_TORCH_PROFILER_DIR if it exists in env, to ensure we don't init Torch Profiler
# This must be done BEFORE any vllm imports to avoid caching env vars.
if "VLLM_TORCH_PROFILER_DIR" in os.environ:
    del os.environ["VLLM_TORCH_PROFILER_DIR"]
    print("Unset VLLM_TORCH_PROFILER_DIR from environment to avoid CUPTI conflict.")

# 1. Import Config classes FIRST
from vllm.config import ParallelConfig

# 2. Apply Monkeypatch IMMEDIATELY
# print("Force disabling Sequence Parallel via monkeypatch (Pre-import)...")
# ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)

# 3. Set Environment Variables for Profiling
# We DO NOT set VLLM_TORCH_PROFILER_DIR to avoid initializing Torch Profiler.
# This prevents CUPTI conflicts.
# We have patched gpu_worker.py to allow profile() calls even if profiler is None.
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./profiler_traces"
# os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
# os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "0"
# os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "0"

# Disable Torch Profiler's CUDA activity to avoid conflict with Nsys
# Nsys uses CUPTI, and Torch Profiler also tries to use CUPTI for CUDA activity.
# This causes CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.
# We only need Torch Profiler to trigger the hooks, not to actually record CUDA events.
# vLLM's gpu_worker.py hardcodes ProfilerActivity.CUDA, so we can't easily disable it via env vars.
# Instead, we will rely on Nsys to attach first, and hope Torch Profiler fails gracefully or we patch it.

# Patched vllm/v1/worker/gpu_worker.py directly to remove ProfilerActivity.CUDA
# This avoids CUPTI conflict with Nsys.
import torch.profiler

# 4. Import torch and vllm core modules AFTER patching and env setup
import torch
from vllm import LLM, SamplingParams

def run_benchmark(config_path):
    print("Loading config from {}".format(config_path))
    
    # Ensure profiler output dir exists
    if not os.path.exists("./profiler_traces"):
        os.makedirs("./profiler_traces")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    parallelism = config.get('parallelism', {})
    data_config = config.get('data', {})

    tp_size = parallelism.get('tp', 1)
    pp_size = parallelism.get('pp', 1)
    ep_size = parallelism.get('ep', 1)
    
    print("Initializing vLLM with TP={}, PP={}, EP={}...".format(tp_size, pp_size, ep_size))

    # Initialize vLLM (Offline Inference)
    # enforce_eager=True is crucial to avoid TorchDynamo errors with DeepSeek-V2
    llm = LLM(
        model=model_config.get('name'),
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        trust_remote_code=True,
        dtype=model_config.get('precision', 'auto'),
        max_model_len=data_config.get('seq_len', 4096),
        enforce_eager=True,
        enable_expert_parallel=(ep_size > 1),
        disable_log_stats=False,
        gpu_memory_utilization=0.9
    )

    # Prepare prompts
    batch_size = data_config.get('batch_size', 1)
    prompt = "The future of artificial intelligence is " * 10
    prompts = [prompt] * batch_size
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=data_config.get('max_tokens', 32),
        ignore_eos=True
    )

    # Warmup
    print("Starting warmup...")
    llm.generate(prompts, sampling_params)

    # Profile run
    print("Starting profile run...")
    
    # Start vLLM internal profiler (triggers workers)
    llm.start_profile()

    profile_iters = config.get('profile_iterations', 1)
    for i in range(profile_iters):
        print("Iteration {}/{}".format(i+1, profile_iters))
        llm.generate(prompts, sampling_params)

    # Stop vLLM internal profiler
    # import time
    # print("Stopping profile run (waiting 2s for sync)...")
    # time.sleep(2)
    # try:
    #     llm.stop_profile()
    # except Exception as e:
    #     print(f"Warning: stop_profile() encountered an error (ignoring): {e}")

    print("Done. Exiting without explicit stop_profile() to avoid shutdown crash.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to configuration YAML")
    args = parser.parse_args()
    run_benchmark(args.config)
