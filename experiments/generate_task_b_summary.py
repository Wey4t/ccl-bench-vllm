import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Data collected from experiments
    data = [
        {
            "experiment": "E2.1_qwen_tp4",
            "model": "Qwen-72B",
            "tp": 4,
            "pp": 1,
            "gpus": 4,
            "ttft_ms": 142.6,
            "tpot_ms": 46.1,
            "bubble_ratio_pct": 0.0, # N/A for PP=1
            "throughput_tokens_sec": 1000 / 46.1 * 1 # Approx based on TPOT (inverse) or just placeholder if not strictly needed
        },
        {
            "experiment": "E2.2_qwen_tp2_pp2",
            "model": "Qwen-72B",
            "tp": 2,
            "pp": 2,
            "gpus": 4,
            "ttft_ms": 290.3,
            "tpot_ms": 48.6,
            "bubble_ratio_pct": 22.91,
            "throughput_tokens_sec": 1000 / 48.6 * 1
        },
        {
            "experiment": "E2.3_qwen_pp4",
            "model": "Qwen-72B",
            "tp": 1,
            "pp": 4,
            "gpus": 4,
            "ttft_ms": 227.9,
            "tpot_ms": 60.4,
            "bubble_ratio_pct": 9.47,
            "throughput_tokens_sec": 1000 / 60.4 * 1
        }
    ]

    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs("experiments", exist_ok=True)
    
    # 1. Save CSV
    csv_path = "experiments/results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    print(df)

    # 2. Generate Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: TTFT Comparison
    ax = axes[0]
    bars = ax.bar(df['experiment'], df['ttft_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('Time to First Token (Lower is Better)')
    ax.bar_label(bars, fmt='%.1f')
    
    # Plot 2: TPOT Comparison
    ax = axes[1]
    bars = ax.bar(df['experiment'], df['tpot_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('Time Per Output Token (Lower is Better)')
    ax.bar_label(bars, fmt='%.1f')
    
    # Plot 3: Bubble Ratio Comparison
    ax = axes[2]
    # Filter out E2.1 for Bubble Ratio as it's PP=1
    df_pp = df[df['pp'] > 1]
    bars = ax.bar(df_pp['experiment'], df_pp['bubble_ratio_pct'], color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Bubble Ratio (%)')
    ax.set_title('Pipeline Bubble Ratio (Lower is Better)')
    ax.bar_label(bars, fmt='%.2f')
    
    plt.tight_layout()
    plot_path = "experiments/scaling_analysis.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
