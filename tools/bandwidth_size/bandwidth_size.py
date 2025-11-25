import json
import os
from typing import Dict, List


def metric_cal(directory: str) -> float:
    """
    Calculate average bandwidth utilization from Kineto traces.

    This metric measures the average bandwidth used by communication operations
    in GB/s across all communication events.

    Args:
        directory (str): Path to the directory containing Kineto trace files.

    Returns:
        float: Average bandwidth in GB/s.
    """

    trace_file = os.path.join(directory, "kineto_trace_0.json")

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Communication event patterns
        comm_patterns = ["nccl", "AllReduce", "AllGather", "ReduceScatter", "AllToAll", "Broadcast"]

        total_bytes = 0
        total_time_us = 0
        comm_event_count = 0

        for event in trace_data.get("traceEvents", []):
            if event.get("cat") != "kernel":
                continue

            name = event.get("name", "").lower()
            dur = event.get("dur", 0)  # duration in microseconds

            if dur == 0:
                continue

            # Check if communication kernel
            if any(pattern.lower() in name for pattern in comm_patterns):
                # Extract size from event args if available
                args = event.get("args", {})
                size_bytes = args.get("size", 0) or args.get("bytes", 0)

                if size_bytes > 0:
                    total_bytes += size_bytes
                    total_time_us += dur
                    comm_event_count += 1

        if total_time_us == 0 or comm_event_count == 0:
            print("Warning: No communication events with size information found")
            return 0.0

        # Calculate bandwidth: (total bytes / total time in seconds) / 1GB
        total_time_sec = total_time_us / 1_000_000
        bandwidth_gbps = (total_bytes / total_time_sec) / (1024**3)

        return bandwidth_gbps

    except FileNotFoundError:
        print(f"File not found: {trace_file}")
        return 0.0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")
        return 0.0
    except Exception as e:
        print(f"Error calculating bandwidth: {e}")
        return 0.0
