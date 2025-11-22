#!/usr/bin/env python3
"""
Analyze and compare results across all experiments
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
        }

        # Calculate throughput
        tokens_per_iter = result['batch_size'] * result['seq_len']
        result['throughput_tokens_sec'] = tokens_per_iter / result['avg_iter_time']

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
        'experiment', 'model', 'tp', 'pp', 'dp', 'ep', 'gpus',
        'throughput_tokens_sec', 'avg_iter_time'
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
