#!/usr/bin/env python3
"""
vLLM Profiling Wrapper with PyTorch ET and Kineto Trace Collection

Usage:
    python vllm_profiler.py --config experiments/configs/E1.1.yaml
"""

import argparse
import json
import os
import torch
from torch.profiler import ExecutionTraceObserver, profile, ProfilerActivity, schedule
from vllm import LLM, SamplingParams
import yaml
import time


class VLLMProfiler:
    def __init__(self, config_path: str):
        """Initialize vLLM profiler with experiment configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Get rank for multi-GPU setup
        self.rank = int(os.environ.get('LOCAL_RANK', 0))

    def create_llm_engine(self):
        """Create vLLM engine with parallelism configuration."""
        model_config = self.config['model']
        parallel_config = self.config['parallelism']

        llm = LLM(
            model=model_config['name'],
            tensor_parallel_size=parallel_config.get('tp', 1),
            pipeline_parallel_size=parallel_config.get('pp', 1),
            trust_remote_code=True,
            dtype=model_config.get('precision', 'bfloat16'),
            max_model_len=self.config['data']['seq_len'],
            gpu_memory_utilization=0.9,
        )

        return llm

    def prepare_prompts(self):
        """Prepare input prompts for profiling."""
        batch_size = self.config['data']['batch_size']
        seq_len = self.config['data']['seq_len']

        # Generate dummy prompts for profiling
        base_prompt = "The future of artificial intelligence is "
        prompts = [base_prompt] * batch_size

        return prompts

    def run_profiled_inference(self):
        """Run vLLM inference with profiling enabled."""
        print(f"[Rank {self.rank}] Starting profiled inference...")

        # Create LLM engine
        llm = self.create_llm_engine()
        prompts = self.prepare_prompts()

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.config['data'].get('max_tokens', 100),
        )

        # Warm-up runs
        warmup_iters = self.config.get('warmup_iterations', 2)
        print(f"[Rank {self.rank}] Running {warmup_iters} warmup iterations...")
        for i in range(warmup_iters):
            _ = llm.generate(prompts, sampling_params)

        # Profiled runs
        profile_iters = self.config.get('profile_iterations', 3)

        # Setup PyTorch ET observer
        et_file = os.path.join(self.output_dir, f"torch_et_{self.rank}.json")
        et = ExecutionTraceObserver()
        et.register_callback(et_file)

        # Kineto trace handler
        def trace_handler(prof):
            kineto_file = os.path.join(self.output_dir, f"kineto_trace_{self.rank}.json")
            prof.export_chrome_trace(kineto_file)
            print(f"[Rank {self.rank}] Saved Kineto trace to {kineto_file}")

        print(f"[Rank {self.rank}] Starting profiled iterations...")

        # Start execution trace observer
        et.start()

        # Profiling context
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=schedule(
                wait=0,
                warmup=0,
                active=profile_iters,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler
        ) as prof:

            iteration_times = []

            for iter_idx in range(profile_iters):
                start_time = time.perf_counter()

                # Run inference
                outputs = llm.generate(prompts, sampling_params)

                end_time = time.perf_counter()
                iter_time = end_time - start_time
                iteration_times.append(iter_time)

                print(f"[Rank {self.rank}] Iteration {iter_idx}: {iter_time:.3f}s")

                # Step profiler
                prof.step()

        # Stop execution trace observer
        et.stop()
        et.unregister_callback()
        print(f"[Rank {self.rank}] Saved PyTorch ET trace to {et_file}")

        # Save iteration timing statistics
        stats = {
            'iteration_times': iteration_times,
            'avg_iteration_time': sum(iteration_times) / len(iteration_times),
            'min_iteration_time': min(iteration_times),
            'max_iteration_time': max(iteration_times),
        }

        stats_file = os.path.join(self.output_dir, f"timing_stats_{self.rank}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"[Rank {self.rank}] Profiling complete!")
        print(f"[Rank {self.rank}] Average iteration time: {stats['avg_iteration_time']:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="vLLM Profiling with Trace Collection")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration YAML")

    args = parser.parse_args()

    profiler = VLLMProfiler(args.config)
    profiler.run_profiled_inference()


if __name__ == "__main__":
    main()
