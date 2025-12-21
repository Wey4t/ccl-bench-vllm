import json
import sys
import numpy as np

def analyze_moe_stats(rank_files):
    stats = {}
    for i, fpath in enumerate(rank_files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                stats[i] = data
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return

    # Analyze the last layer or a specific layer
    layer_name = "model.layers.25.mlp" 
    
    print(f"Analyzing {layer_name}...")
    
    # Assume 64 experts, 4 ranks, linear assignment
    num_experts = 64
    num_ranks = 4
    experts_per_rank = num_experts // num_ranks
    
    for rank, data in stats.items():
        if layer_name not in data:
            print(f"Rank {rank}: Layer {layer_name} not found.")
            continue
            
        layer_stats = data[layer_name]
        counts = layer_stats['expert_counts']
        
        # Calculate Local Load
        start_expert = rank * experts_per_rank
        end_expert = start_expert + experts_per_rank
        
        local_counts = counts[start_expert:end_expert]
        local_load = sum(local_counts)
        
        print(f"Rank {rank} (Experts {start_expert}-{end_expert-1}):")
        print(f"  Local Load (Tokens): {local_load}")
        print(f"  Max Expert Load: {max(local_counts) if local_counts else 0}")
        print(f"  Active Local Experts: {sum(1 for c in local_counts if c > 0)}")

if __name__ == "__main__":
    files = [
        "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/moe_stats_rank0.json",
        "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/moe_stats_rank1.json",
        "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/moe_stats_rank2.json",
        "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/moe_stats_rank3.json"
    ]
    analyze_moe_stats(files)
