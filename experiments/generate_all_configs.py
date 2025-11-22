#!/usr/bin/env python3
"""
Generate all experiment configuration files for E1.1-E3.4
"""

import os
import yaml


EXPERIMENTS = {
    # E1: Llama-8B experiments
    "E1.1": {
        "description": "Baseline inference with Llama-8B on 1 GPU",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 1,
        "batch_size": 4,
    },
    "E1.2": {
        "description": "Llama-8B with Tensor Parallelism TP=2",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 2, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 2,
        "batch_size": 4,
    },
    "E1.3": {
        "description": "Llama-8B with Tensor Parallelism TP=4",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 4, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 4,
        "batch_size": 4,
    },
    "E1.4": {
        "description": "Llama-8B with Data Parallelism DP=2",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 2, "ep": 1},
        "gpus": 2,
        "batch_size": 4,
    },
    "E1.6": {
        "description": "Llama-8B with Pipeline Parallelism PP=2",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 1, "pp": 2, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 2,
        "batch_size": 4,
    },

    # E2: Multi-parallelism experiments
    "E2.1": {
        "description": "Llama-8B with TP=2, DP=2 on 4 GPUs",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 2, "pp": 1, "dp_replicate": 1, "dp_shard": 2, "ep": 1},
        "gpus": 4,
        "batch_size": 4,
    },
    "E2.2": {
        "description": "Llama-8B with TP=2, PP=2 on 4 GPUs",
        "model": "meta-llama/Llama-3.1-8B",
        "parallelism": {"tp": 2, "pp": 2, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 4,
        "batch_size": 4,
    },
    "E2.3": {
        "description": "Qwen-32B with TP=4 on 4 GPUs",
        "model": "Qwen/Qwen-32B",
        "parallelism": {"tp": 4, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 4,
        "batch_size": 2,
    },

    # E3: Expert Parallelism (MoE) experiments
    "E3.1": {
        "description": "DeepSeek-V2-Lite MoE on 1 GPU",
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 1},
        "gpus": 1,
        "batch_size": 2,
    },
    "E3.2": {
        "description": "DeepSeek-V2-Lite with EP=2",
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 2},
        "gpus": 2,
        "batch_size": 2,
    },
    "E3.3": {
        "description": "DeepSeek-V2-Lite with EP=4",
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "parallelism": {"tp": 1, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 4},
        "gpus": 4,
        "batch_size": 2,
    },
    "E3.4": {
        "description": "DeepSeek-V2-Lite with TP=2, EP=2",
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "parallelism": {"tp": 2, "pp": 1, "dp_replicate": 1, "dp_shard": 1, "ep": 2},
        "gpus": 4,
        "batch_size": 2,
    },
}


def generate_config(exp_id, exp_info):
    """Generate configuration file for an experiment."""
    config = {
        "experiment_id": exp_id,
        "description": exp_info["description"],
        "model": {
            "name": exp_info["model"],
            "precision": "bfloat16",
        },
        "parallelism": exp_info["parallelism"],
        "data": {
            "batch_size": exp_info["batch_size"],
            "seq_len": 8192,
            "max_tokens": 128,
        },
        "warmup_iterations": 2,
        "profile_iterations": 5,
        "output_dir": f"trace_collection/{exp_info['model'].split('/')[-1].lower()}-vllm-perlmutter-{exp_id}",
    }

    return config


def main():
    os.makedirs("experiments/configs", exist_ok=True)

    for exp_id, exp_info in EXPERIMENTS.items():
        config = generate_config(exp_id, exp_info)

        filename = f"experiments/configs/{exp_id}_{exp_info['model'].split('/')[-1].lower()}.yaml"

        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Generated {filename}")

    print(f"\nTotal experiments generated: {len(EXPERIMENTS)}")


if __name__ == "__main__":
    main()
