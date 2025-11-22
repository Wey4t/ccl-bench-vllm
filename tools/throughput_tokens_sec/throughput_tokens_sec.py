import json
import os
from typing import Dict


def metric_cal(directory: str) -> float:
    """
    Calculate throughput in tokens per second from timing stats.

    Args:
        directory (str): Path to the directory containing timing_stats JSON files.

    Returns:
        float: Throughput in tokens per second.
    """

    # Read workload card to get batch size and sequence length
    workload_card_path = os.path.join(directory, "workload_card.yaml")
    timing_stats_path = os.path.join(directory, "timing_stats_0.json")

    try:
        # Load timing statistics
        with open(timing_stats_path, 'r') as f:
            timing_stats = json.load(f)

        avg_iter_time = timing_stats['avg_iteration_time']

        # Load workload card for batch size and sequence length
        import yaml
        with open(workload_card_path, 'r') as f:
            workload_card = yaml.safe_load(f)

        batch_size = workload_card['workload']['data']['batch_size']
        seq_len = workload_card['workload']['data']['seq_len']

        # Calculate tokens per iteration
        tokens_per_iter = batch_size * seq_len

        # Calculate throughput
        throughput = tokens_per_iter / avg_iter_time

        return throughput

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 0.0
    except Exception as e:
        print(f"Error calculating throughput: {e}")
        return 0.0
