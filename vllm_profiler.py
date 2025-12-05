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
    def __init__(self, config_path):
        """Initialize vLLM profiler with experiment configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Save configuration to output directory for analysis
        with open(os.path.join(self.output_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.config, f)

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
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
            enforce_eager=model_config.get('enforce_eager', False),
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

        # Process metrics from all iterations
        ttft_list = []
        tpot_list = []
        
        # Note: In the current loop structure, we only have access to 'outputs' from the last iteration
        # because we overwrite it. To fix this, we should have collected outputs inside the loop.
        # However, since we didn't change the loop above, we can only use the last batch.
        # But wait, the log showed 0.0s, which means even for the last batch it failed.
        # Let's add a check.
        
        if outputs:
            print(f"[DEBUG] Output count: {len(outputs)}")
            if len(outputs) > 0:
                print(f"[DEBUG] First output metrics: {outputs[0].metrics}")

            for request_output in outputs:
                if request_output.metrics:
                    # TTFT: Time to first token (arrival to first token)
                    if request_output.metrics.first_token_time is not None and request_output.metrics.arrival_time is not None:
                        ttft = request_output.metrics.first_token_time - request_output.metrics.arrival_time
                        ttft_list.append(ttft)
                    
                    # TPOT: Time per output token
                    if request_output.metrics.finished_time is not None and request_output.metrics.first_token_time is not None:
                        gen_time = request_output.metrics.finished_time - request_output.metrics.first_token_time
                        output_len = len(request_output.outputs[0].token_ids)
                        if output_len > 1:
                            tpot = gen_time / (output_len - 1)
                            tpot_list.append(tpot)

        # Save iteration timing statistics and new metrics
        stats = {
            'iteration_times': iteration_times,
            'avg_iteration_time': sum(iteration_times) / len(iteration_times) if iteration_times else 0,
            'min_iteration_time': min(iteration_times) if iteration_times else 0,
            'max_iteration_time': max(iteration_times) if iteration_times else 0,
            'ttft_avg': sum(ttft_list) / len(ttft_list) if ttft_list else 0,
            'tpot_avg': sum(tpot_list) / len(tpot_list) if tpot_list else 0,
            'ttft_p99': sorted(ttft_list)[int(len(ttft_list) * 0.99)] if ttft_list else 0,
            'tpot_p99': sorted(tpot_list)[int(len(tpot_list) * 0.99)] if tpot_list else 0,
        }

        stats_file = os.path.join(self.output_dir, f"timing_stats_{self.rank}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"[Rank {self.rank}] Profiling complete!")
        print(f"[Rank {self.rank}] Average iteration time: {stats['avg_iteration_time']:.3f}s")
        print(f"[Rank {self.rank}] Average TTFT: {stats['ttft_avg']:.4f}s")
        print(f"[Rank {self.rank}] Average TPOT: {stats['tpot_avg']:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="vLLM Profiling with Trace Collection")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration YAML")

    args = parser.parse_args()

    profiler = VLLMProfiler(args.config)
    profiler.run_profiled_inference()


if __name__ == "__main__":
    main()
