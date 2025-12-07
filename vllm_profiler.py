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

        print("[DEBUG] Initializing LLM with disable_log_stats=False")
        llm = LLM(
            model=model_config['name'],
            tensor_parallel_size=parallel_config.get('tp', 1),
            pipeline_parallel_size=parallel_config.get('pp', 1),
            trust_remote_code=True,
            dtype=model_config.get('precision', 'bfloat16'),
            max_model_len=self.config['data']['seq_len'],
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
            enforce_eager=model_config.get('enforce_eager', False),
            disable_log_stats=False,
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
        print("[Rank {}] Starting profiled inference...".format(self.rank))

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
        print("[Rank {}] Running {} warmup iterations...".format(self.rank, warmup_iters))
        for i in range(warmup_iters):
            _ = llm.generate(prompts, sampling_params)

        # Inspect LLM engine internals
        print("[DEBUG] LLM Engine dir: {}".format(dir(llm.llm_engine)))
        if hasattr(llm.llm_engine, 'stat_logger'):
             print("[DEBUG] Stat Logger: {}".format(llm.llm_engine.stat_logger))
             print("[DEBUG] Stat Logger dir: {}".format(dir(llm.llm_engine.stat_logger)))

        # Profiled runs
        profile_iters = self.config.get('profile_iterations', 3)

        # Setup PyTorch ET observer
        et_file = os.path.join(self.output_dir, "torch_et_{}.json".format(self.rank))
        et = ExecutionTraceObserver()
        et.register_callback(et_file)

        # Kineto trace handler
        def trace_handler(prof):
            kineto_file = os.path.join(self.output_dir, "kineto_trace_{}.json".format(self.rank))
            prof.export_chrome_trace(kineto_file)
            print("[Rank {}] Saved Kineto trace to {}".format(self.rank, kineto_file))

        print("[Rank {}] Starting profiled iterations...".format(self.rank))

        iteration_times = []
        outputs = None

        # Start execution trace observer
        et.start()

        # Kineto profiler context
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=schedule(wait=0, warmup=0, active=profile_iters),
            record_shapes=False,
            on_trace_ready=trace_handler,
        ) as prof:
            for iter_idx in range(profile_iters):
                start_time = time.perf_counter()

                # Run inference and ask vLLM to return metrics
                outputs = llm.generate(prompts, sampling_params, collect_metrics=True)

                end_time = time.perf_counter()
                iter_time = end_time - start_time
                iteration_times.append(iter_time)

                prof.step()

                print("[Rank {}] Iteration {}: {:.3f}s".format(self.rank, iter_idx, iter_time))

        # Stop execution trace observer
        et.stop()
        et.unregister_callback()
        print("[Rank {}] Saved PyTorch ET trace to {}".format(self.rank, et_file))

        # Process metrics from all iterations
        ttft_list = []
        tpot_list = []
        
        # Try to extract from StatLogger if RequestOutput metrics are missing
        if outputs and outputs[0].metrics is None and hasattr(llm.llm_engine, 'stat_logger'):
            print("[DEBUG] Attempting to extract metrics from StatLogger...")
            logger = llm.llm_engine.stat_logger
            # Inspect histogram internals
            if hasattr(logger, 'histogram_time_to_first_token'):
                h_ttft = logger.histogram_time_to_first_token
                # Handle dict structure {model_id: Histogram}
                if isinstance(h_ttft, dict):
                    for _, histogram in h_ttft.items():
                        if hasattr(histogram, 'collect'):
                            metrics_data = histogram.collect()
                            if metrics_data and len(metrics_data) > 0:
                                samples = metrics_data[0].samples
                                sum_val = next((s.value for s in samples if s.name.endswith('_sum')), 0)
                                count_val = next((s.value for s in samples if s.name.endswith('_count')), 0)
                                if count_val > 0:
                                    avg_ttft = sum_val / count_val
                                    print("[DEBUG] Extracted Avg TTFT from Histogram: {}s".format(avg_ttft))
                                    ttft_list.extend([avg_ttft] * int(count_val))

            if hasattr(logger, 'histogram_time_per_output_token'):
                h_tpot = logger.histogram_time_per_output_token
                if isinstance(h_tpot, dict):
                    for _, histogram in h_tpot.items():
                        if hasattr(histogram, 'collect'):
                            metrics_data = histogram.collect()
                            if metrics_data and len(metrics_data) > 0:
                                samples = metrics_data[0].samples
                                sum_val = next((s.value for s in samples if s.name.endswith('_sum')), 0)
                                count_val = next((s.value for s in samples if s.name.endswith('_count')), 0)
                                if count_val > 0:
                                    avg_tpot = sum_val / count_val
                                    print("[DEBUG] Extracted Avg TPOT from Histogram: {}s".format(avg_tpot))
                                    tpot_list.extend([avg_tpot] * int(count_val))

        if outputs:
            print("[DEBUG] Output count: {}".format(len(outputs)))
            if len(outputs) > 0:
                print("[DEBUG] First output type: {}".format(type(outputs[0])))
                print("[DEBUG] First output dir: {}".format(dir(outputs[0])))
                print("[DEBUG] First output: {}".format(outputs[0]))
                try:
                    print("[DEBUG] First output metrics: {}".format(outputs[0].metrics))
                except AttributeError:
                    print("[DEBUG] First output has no metrics attribute")

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
                    print("[DEBUG] Object has no 'metrics' attribute")

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

        stats_file = os.path.join(self.output_dir, "timing_stats_{}.json".format(self.rank))
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("[Rank {}] Profiling complete!".format(self.rank))
        print("[Rank {}] Average iteration time: {:.3f}s".format(self.rank, stats['avg_iteration_time']))
        print("[Rank {}] Average TTFT: {:.4f}s".format(self.rank, stats['ttft_avg']))
        print("[Rank {}] Average TPOT: {:.4f}s".format(self.rank, stats['tpot_avg']))


def main():
    parser = argparse.ArgumentParser(description="vLLM Profiling with Trace Collection")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration YAML")

    args = parser.parse_args()

    profiler = VLLMProfiler(args.config)
    profiler.run_profiled_inference()


if __name__ == "__main__":
    main()
