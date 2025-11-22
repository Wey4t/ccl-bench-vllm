import json
import os


def metric_cal(directory: str) -> float:
    """
    Calculate average iteration wall-clock time.

    Args:
        directory (str): Path to the directory containing timing_stats JSON files.

    Returns:
        float: Average iteration wall-clock time in seconds.
    """

    timing_stats_path = os.path.join(directory, "timing_stats_0.json")

    try:
        with open(timing_stats_path, 'r') as f:
            timing_stats = json.load(f)

        return timing_stats['avg_iteration_time']

    except FileNotFoundError:
        print(f"File not found: {timing_stats_path}")
        return 0.0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {timing_stats_path}")
        return 0.0
