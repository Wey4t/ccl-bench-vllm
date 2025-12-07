import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import re

def parse_layer_id(layer_name):
    # Try to extract number from string like "model.layers.10.mlp"
    match = re.search(r'layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return -1

def plot_expert_stats(input_dir, output_dir):
    print(f"Scanning directory: {input_dir}")
    files = glob.glob(os.path.join(input_dir, "moe_stats_rank*.json"))
    
    if not files:
        print("No moe_stats files found!")
        return

    print(f"Found {len(files)} files. Processing...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate data by rank and layer
    # Structure: data[rank][layer_name] = {'counts': [], 'calls': []}
    data_by_rank = {}
    
    for fpath in files:
        try:
            # Extract rank from filename
            filename = os.path.basename(fpath)
            rank_match = re.search(r'rank(\d+)', filename)
            if not rank_match:
                print(f"Skipping {filename}: cannot parse rank")
                continue
            rank = int(rank_match.group(1))
            
            with open(fpath, 'r') as f:
                file_data = json.load(f)
                
            data_by_rank[rank] = file_data
            
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # Get all unique layers sorted by ID
    all_layers = set()
    for rank_data in data_by_rank.values():
        all_layers.update(rank_data.keys())
    
    sorted_layers = sorted(list(all_layers), key=parse_layer_id)
    
    # --- Summary Data Collection ---
    # Logic:
    # 1. Aggregate counts from ALL ranks to get "Global Expert Counts" for each layer.
    #    (How many tokens across the entire cluster wanted to go to Expert X)
    # 2. Map Experts to GPUs based on simple partitioning.
    #    (Expert 0-15 -> GPU 0, etc.)
    # 3. Calculate "True GPU Load" based on this mapping.
    
    ranks = sorted(data_by_rank.keys())
    num_ranks = len(ranks)
    
    # Storage for global stats
    # global_layer_stats[layer_id] = {'counts': [count_expert_0, ...], 'calls': [call_expert_0, ...]}
    global_layer_stats = {}
    
    # Storage for True GPU Load
    # true_gpu_load[rank] = {'tokens': 0, 'calls': 0}
    true_gpu_load = {r: {'tokens': 0, 'calls': 0} for r in ranks}
    
    # Storage for Layer Imbalance
    layer_imbalance = {} # layer_id -> {'cv': float, 'max_load': int, 'min_load': int}
    
    # Storage for Total Model Expert Counts
    total_model_expert_counts = None
    total_model_expert_calls = None

    for layer_name in sorted_layers:
        layer_id = parse_layer_id(layer_name)
        
        # 1. Calculate Global Expert Counts for this layer
        # Initialize with zeros based on the first rank's data size
        first_rank_data = data_by_rank[ranks[0]][layer_name]
        num_experts = len(first_rank_data['expert_counts'])
        
        global_counts = np.zeros(num_experts)
        global_calls = np.zeros(num_experts)
        
        for rank in ranks:
            if layer_name in data_by_rank[rank]:
                global_counts += np.array(data_by_rank[rank][layer_name]['expert_counts'])
                # For calls, it's tricky. 
                # If Rank 0 calls Expert A in Batch 1, and Rank 1 calls Expert A in Batch 1.
                # Is that 1 call or 2 calls?
                # From a compute perspective, Expert A has to run Batch 1.
                # Since we don't have batch-level granularity here, we can sum them as an approximation of "Total Active Batches processed by Expert A"
                global_calls += np.array(data_by_rank[rank][layer_name]['expert_active_calls'])

        global_layer_stats[layer_id] = {'counts': global_counts, 'calls': global_calls}
        
        # Accumulate for Total Model Plot
        if total_model_expert_counts is None:
            total_model_expert_counts = np.zeros_like(global_counts)
            total_model_expert_calls = np.zeros_like(global_calls)
        
        if len(global_counts) == len(total_model_expert_counts):
            total_model_expert_counts += global_counts
            total_model_expert_calls += global_calls
        else:
            print(f"Warning: Layer {layer_id} has different number of experts ({len(global_counts)}) than expected ({len(total_model_expert_counts)}). Skipping accumulation for total plot.")
        
        # 2. Calculate Imbalance (CV) for this layer based on Global Counts
        if np.sum(global_counts) > 0:
            mean = np.mean(global_counts)
            std = np.std(global_counts)
            cv = std / mean if mean > 0 else 0
        else:
            cv = 0
        
        layer_imbalance[layer_id] = {
            'cv': cv,
            'max_load': np.max(global_counts),
            'min_load': np.min(global_counts)
        }

        # 3. Map to GPUs (True Load Calculation)
        experts_per_rank = num_experts // num_ranks
        
        for i, rank in enumerate(ranks):
            start_idx = i * experts_per_rank
            end_idx = (i + 1) * experts_per_rank
            
            # Handle last rank taking remainder if any (though usually divisible)
            if i == num_ranks - 1:
                end_idx = num_experts
                
            rank_expert_counts = global_counts[start_idx:end_idx]
            rank_expert_calls = global_calls[start_idx:end_idx]
            
            true_gpu_load[rank]['tokens'] += np.sum(rank_expert_counts)
            true_gpu_load[rank]['calls'] += np.sum(rank_expert_calls)

    # --- Plot 1: True GPU Load Summary ---
    print("Plotting True GPU Load Summary...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    gpu_tokens = [true_gpu_load[r]['tokens'] for r in ranks]
    gpu_calls = [true_gpu_load[r]['calls'] for r in ranks]
    
    # Plot Tokens
    bars1 = ax1.bar(ranks, gpu_tokens, color='skyblue')
    ax1.set_title('True Compute Load per GPU (Total Tokens)')
    ax1.set_xlabel('GPU Rank')
    ax1.set_ylabel('Total Tokens Processed')
    ax1.set_xticks(ranks)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels and percentage deviation
    mean_tokens = np.mean(gpu_tokens)
    for bar in bars1:
        height = bar.get_height()
        diff_pct = ((height - mean_tokens) / mean_tokens) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({diff_pct:+.1f}%)',
                ha='center', va='bottom')
    
    # Plot Calls
    bars2 = ax2.bar(ranks, gpu_calls, color='lightgreen')
    ax2.set_title('True Active Batches per GPU (Total Calls)')
    ax2.set_xlabel('GPU Rank')
    ax2.set_ylabel('Total Active Batches')
    ax2.set_xticks(ranks)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_true_gpu_load.png"))
    plt.close()

    # --- Plot 2: Layer Imbalance Summary ---
    print("Plotting Layer Imbalance Summary...")
    plt.figure(figsize=(12, 6))
    
    sorted_layer_ids = sorted(layer_imbalance.keys())
    cvs = [layer_imbalance[lid]['cv'] for lid in sorted_layer_ids]
    
    plt.plot(sorted_layer_ids, cvs, marker='o', color='purple', linewidth=2)
    plt.title('Global Expert Load Imbalance (CV) per Layer')
    plt.xlabel('Layer ID')
    plt.ylabel('Coefficient of Variation (Std/Mean)')
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted_layer_ids)
    
    # Add mean line
    mean_cv = np.mean(cvs)
    plt.axhline(y=mean_cv, color='r', linestyle='--', label=f'Mean CV: {mean_cv:.2f}')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "summary_layer_imbalance.png"))
    plt.close()

    # --- Plot 3: Global Expert Distribution (Heatmap style) ---
    # Optional: Visualize load across all experts and layers
    
    # --- Existing Per-Layer Plots (Updated to show Global Counts) ---
    for layer_name in sorted_layers:
        layer_id = parse_layer_id(layer_name)
        # print(f"Plotting Layer {layer_id} ({layer_name})...")
        
        global_counts = global_layer_stats[layer_id]['counts']
        global_calls = global_layer_stats[layer_id]['calls']
        num_experts = len(global_counts)
        experts = np.arange(num_experts)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Color bars by GPU ownership
        experts_per_rank = num_experts // num_ranks
        colors = []
        rank_labels = []
        
        for i in range(num_experts):
            rank_owner = min(i // experts_per_rank, num_ranks - 1)
            denominator = max(num_ranks - 1, 1)
            colors.append(plt.cm.viridis(rank_owner / denominator))
            
        # Plot Tokens
        ax1.bar(experts, global_counts, color=colors, alpha=0.8)
        
        # Create custom legend
        from matplotlib.patches import Patch
        denominator = max(num_ranks - 1, 1)
        legend_elements = [Patch(facecolor=plt.cm.viridis(i / denominator), 
                               label=f'GPU {i}') for i in ranks]
        
        ax1.set_title(f'Layer {layer_id}: Global Token Distribution per Expert')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Total Tokens (Global)')
        ax1.legend(handles=legend_elements, title="Hosted on")
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xlim(-0.5, num_experts + 0.5)

        # Plot Calls
        ax2.bar(experts, global_calls, color=colors, alpha=0.8)
        
        ax2.set_title(f'Layer {layer_id}: Global Active Calls per Expert')
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Total Active Calls (Global)')
        ax2.legend(handles=legend_elements, title="Hosted on")
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xlim(-0.5, num_experts + 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{layer_id:02d}_global_stats.png"))
        plt.close()

    # --- Plot 4: Total Model Expert Distribution ---
    if total_model_expert_counts is not None:
        print("Plotting Total Model Expert Distribution...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        num_experts = len(total_model_expert_counts)
        experts = np.arange(num_experts)
        experts_per_rank = num_experts // num_ranks
        
        colors = []
        for i in range(num_experts):
            rank_owner = min(i // experts_per_rank, num_ranks - 1)
            denominator = max(num_ranks - 1, 1)
            colors.append(plt.cm.viridis(rank_owner / denominator))
            
        # Plot Tokens
        ax1.bar(experts, total_model_expert_counts, color=colors, alpha=0.8)
        
        # Legend
        from matplotlib.patches import Patch
        denominator = max(num_ranks - 1, 1)
        legend_elements = [Patch(facecolor=plt.cm.viridis(i / denominator), 
                               label=f'GPU {i}') for i in ranks]
        
        ax1.set_title('Total Model: Global Token Distribution per Expert (All Layers Summed)')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Total Tokens (All Layers)')
        ax1.legend(handles=legend_elements, title="Hosted on")
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xlim(-0.5, num_experts + 0.5)

        # Plot Calls
        ax2.bar(experts, total_model_expert_calls, color=colors, alpha=0.8)
        
        ax2.set_title('Total Model: Global Active Calls per Expert (All Layers Summed)')
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Total Active Calls (All Layers)')
        ax2.legend(handles=legend_elements, title="Hosted on")
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xlim(-0.5, num_experts + 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary_total_model_expert_dist.png"))
        plt.close()

    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MoE Expert Statistics")
    parser.add_argument("input_dir", help="Directory containing moe_stats_rank*.json files")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    plot_expert_stats(args.input_dir, args.output_dir)
