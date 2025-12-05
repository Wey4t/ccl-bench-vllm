#!/usr/bin/env python3
"""
Analyze and compare results across all experiments
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path to allow imports if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tools.trace_analyzer import TraceAnalyzer
    from tools.ttft.ttft import metric_cal as cal_ttft
    from tools.tpot.tpot import metric_cal as cal_tpot
    from tools.pipeline_bubble_ratio.pipeline_bubble_ratio import metric_cal as cal_bubble_ratio
except ImportError:
    print("Warning: Could not import tools")
    TraceAnalyzer = None


def collect_metrics(trace_collection_dir):
    """Collect metrics from all experiment traces."""

    results = []

    for trace_dir in Path(trace_collection_dir).iterdir():
        if not trace_dir.is_dir():
            continue

        # Read workload card
        workload_card_path = trace_dir / "workload_card.yaml"
        if not workload_card_path.exists():
            continue

        with open(workload_card_path, 'r') as f:
            workload_card = yaml.safe_load(f)

        # Read timing stats
        timing_stats_path = trace_dir / "timing_stats_0.json"
        if not timing_stats_path.exists():
            continue

        with open(timing_stats_path, 'r') as f:
            timing_stats = json.load(f)

        # Extract key information
        parallelism = workload_card['Model-executor']['model_plan_parallelization']
        
        # New metrics using tools
        ttft_ms = 0.0
        tpot_ms = 0.0
        bubble_ratio = 0.0
        
        try:
            ttft_ms = cal_ttft(str(trace_dir))
            tpot_ms = cal_tpot(str(trace_dir))
            bubble_ratio = cal_bubble_ratio(str(trace_dir))
        except Exception as e:
            print(f"Error calculating metrics for {trace_dir.name}: {e}")

        # Analyze traces for other metrics (Comm Overhead, MFU)
        comm_overhead = 0.0
        sm_efficiency = 0.0
        
        if TraceAnalyzer:
            kineto_trace_path = trace_dir / f"kineto_trace_0.json"
            if kineto_trace_path.exists():
                analyzer = TraceAnalyzer(str(kineto_trace_path))
                comm_overhead = analyzer.calculate_comm_overhead()
                # bubble_ratio is already calculated via tool
                sm_efficiency = analyzer.calculate_sm_efficiency()

        result = {
            'experiment': trace_dir.name,
            'model': workload_card['workload']['model']['model_family'],
            'tp': parallelism['tp'],
            'pp': parallelism['pp'],
            'dp': parallelism['dp_shard'],
            'ep': parallelism.get('ep', 1),
            'gpus': workload_card['workload']['hardware']['xpu_spec']['total_count'],
            'batch_size': workload_card['workload']['data']['batch_size'],
            'seq_len': workload_card['workload']['data']['seq_len'],
            'avg_iter_time': timing_stats['avg_iteration_time'],
            'min_iter_time': timing_stats['min_iteration_time'],
            'max_iter_time': timing_stats['max_iteration_time'],
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'comm_overhead_pct': comm_overhead,
            'bubble_ratio_pct': bubble_ratio,
            'sm_efficiency_pct': sm_efficiency,
        }

        # Calculate throughput
        tokens_per_iter = result['batch_size'] * result['seq_len']
        result['throughput_tokens_sec'] = tokens_per_iter / result['avg_iter_time']
        
        # Calculate MFU (Approximate)
        # MFU = (Throughput * FLOPs_per_token) / (Num_GPUs * Peak_FLOPs_per_GPU)
        # Approx FLOPs per token = 2 * Num_Params
        # We need model size in params. Let's estimate from model name or config.
        # For now, let's assume Llama-8B has ~8B params.
        model_name = result['model'].lower()
        num_params = 8e9 # Default to 8B
        if '70b' in model_name:
            num_params = 70e9
        elif 'moe' in model_name or 'deepseek' in model_name:
            # For MoE, active params is what matters for FLOPs
            num_params = 13e9 # Approx active params for some MoEs
            
        flops_per_token = 2 * num_params
        # Peak FLOPs for A100 BF16 ~ 312 TFLOPS
        peak_flops_per_gpu = 312e12 
        
        total_flops = result['throughput_tokens_sec'] * flops_per_token
        total_peak_flops = result['gpus'] * peak_flops_per_gpu
        
        result['mfu_pct'] = (total_flops / total_peak_flops) * 100

        results.append(result)

    return pd.DataFrame(results)


def plot_scaling_analysis(df):
    """Generate scaling analysis plots."""

    # Plot 1: Throughput vs Number of GPUs
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Throughput scaling
    ax = axes[0, 0]
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax.plot(model_df['gpus'], model_df['throughput_tokens_sec'],
                marker='o', label=model)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput Scaling')
    ax.legend()
    ax.grid(True)

    # Iteration time comparison
    ax = axes[0, 1]
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        ax.plot(model_df['gpus'], model_df['avg_iter_time'],
                marker='s', label=model)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Iteration Time (s)')
    ax.set_title('Iteration Time vs GPUs')
    ax.legend()
    ax.grid(True)

    # TP scaling efficiency
    ax = axes[1, 0]
    tp_df = df[df['pp'] == 1]
    for model in tp_df['model'].unique():
        model_df = tp_df[tp_df['model'] == model]
        baseline = model_df[model_df['tp'] == 1]['throughput_tokens_sec'].values
        if len(baseline) > 0:
            efficiency = (model_df['throughput_tokens_sec'] / baseline[0]) / model_df['tp'] * 100
            ax.plot(model_df['tp'], efficiency, marker='o', label=model)
    ax.set_xlabel('Tensor Parallelism (TP)')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('TP Scaling Efficiency')
    ax.axhline(y=100, color='r', linestyle='--', label='Ideal')
    ax.legend()
    ax.grid(True)

    # EP scaling for MoE models
    ax = axes[1, 1]
    ep_df = df[df['ep'] > 1]
    if len(ep_df) > 0:
        for model in ep_df['model'].unique():
            model_df = ep_df[ep_df['model'] == model]
            ax.plot(model_df['ep'], model_df['throughput_tokens_sec'],
                    marker='d', label=model)
        ax.set_xlabel('Expert Parallelism (EP)')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title('EP Scaling for MoE Models')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('experiments/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved scaling analysis plot to experiments/scaling_analysis.png")


def generate_summary_table(df):
    """Generate summary table for all experiments."""

    summary_cols = [
        'experiment', 'model', 'tp', 'pp', 'ep', 'gpus',
        'throughput_tokens_sec', 'ttft_ms', 'tpot_ms', 
        'mfu_pct', 'comm_overhead_pct', 'bubble_ratio_pct'
    ]

    summary = df[summary_cols].copy()
    summary = summary.sort_values(['model', 'gpus'])

    # Save to CSV
    summary.to_csv('experiments/results_summary.csv', index=False)
    print("\nResults Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to experiments/results_summary.csv")

    return summary


def main():
    trace_collection_dir = "trace_collection"

    if not os.path.exists(trace_collection_dir):
        print(f"Error: {trace_collection_dir} not found")
        return

    print("Collecting metrics from experiments...")
    df = collect_metrics(trace_collection_dir)

    if df.empty:
        print("No experiment results found!")
        return

    print(f"\nFound {len(df)} experiments")

    # Generate summary table
    summary = generate_summary_table(df)

    # Generate plots
    print("\nGenerating scaling analysis plots...")
    plot_scaling_analysis(df)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
