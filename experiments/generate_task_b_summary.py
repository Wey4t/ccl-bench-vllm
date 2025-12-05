import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Constants for MFU calculation
    # Note: Config says Qwen/Qwen2.5-32B, so we use ~32.5B params
    NUM_PARAMS = 32.5e9 
    FLOPS_PER_TOKEN = 2 * NUM_PARAMS
    NUM_GPUS = 4
    PEAK_FLOPS_PER_GPU = 312e12  # A100 BF16 Tensor Core Peak
    BATCH_SIZE = 16 # From config

    # Data collected from experiments
    data = [
        {
            "experiment": "E2.1_qwen_tp4",
            "config": "TP=4, PP=1",
            "ttft_ms": 142.6,
            "tpot_ms": 46.1,
            "bubble_ratio_pct": 0.0, 
        },
        {
            "experiment": "E2.2_qwen_tp2_pp2",
            "config": "TP=2, PP=2",
            "ttft_ms": 290.3,
            "tpot_ms": 48.6,
            "bubble_ratio_pct": 22.91,
        },
        {
            "experiment": "E2.3_qwen_pp4",
            "config": "TP=1, PP=4",
            "ttft_ms": 227.9,
            "tpot_ms": 60.4,
            "bubble_ratio_pct": 9.47,
        }
    ]

    # Calculate derived metrics
    for d in data:
        # System Throughput (tokens/sec) ~= Batch_Size / TPOT (s)
        # TPOT is in ms, so / 1000
        tpot_sec = d['tpot_ms'] / 1000.0
        throughput = BATCH_SIZE / tpot_sec
        d['throughput_tokens_sec'] = throughput
        
        # MFU = (Throughput * FLOPs/token) / (Num_GPUs * Peak_FLOPs/GPU)
        total_flops = throughput * FLOPS_PER_TOKEN
        total_peak = NUM_GPUS * PEAK_FLOPS_PER_GPU
        d['mfu_pct'] = (total_flops / total_peak) * 100

    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs("experiments", exist_ok=True)
    
    # 1. Save CSV
    csv_path = "experiments/results_summary.csv"
    # Reorder columns
    cols = ["experiment", "config", "ttft_ms", "tpot_ms", "bubble_ratio_pct", "throughput_tokens_sec", "mfu_pct"]
    df = df[cols]
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    print(df.to_string())

    # 2. Generate Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TTFT
    ax = axes[0, 0]
    bars = ax.bar(df['config'], df['ttft_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('Time to First Token (Lower is Better)')
    ax.bar_label(bars, fmt='%.1f')
    
    # Plot 2: TPOT
    ax = axes[0, 1]
    bars = ax.bar(df['config'], df['tpot_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('Time Per Output Token (Lower is Better)')
    ax.bar_label(bars, fmt='%.1f')
    
    # Plot 3: Bubble Ratio
    ax = axes[1, 0]
    # Filter out E2.1 for Bubble Ratio
    df_pp = df[df['bubble_ratio_pct'] > 0]
    if not df_pp.empty:
        bars = ax.bar(df_pp['config'], df_pp['bubble_ratio_pct'], color=['#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Bubble Ratio (%)')
        ax.set_title('Pipeline Bubble Ratio (Lower is Better)')
        ax.bar_label(bars, fmt='%.2f')
    else:
        ax.text(0.5, 0.5, "No Pipeline Parallelism", ha='center', va='center')
        
    # Plot 4: MFU
    ax = axes[1, 1]
    bars = ax.bar(df['config'], df['mfu_pct'], color=['#9467bd', '#8c564b', '#e377c2'])
    ax.set_ylabel('MFU (%)')
    ax.set_title('Model FLOPs Utilization (Higher is Better)')
    ax.bar_label(bars, fmt='%.2f')
    
    plt.tight_layout()
    plot_path = "experiments/scaling_analysis.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
