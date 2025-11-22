import json
import os
from typing import List, Tuple


def metric_cal(directory: str) -> float:
    """
    Calculate communication-computation overlap percentage from Kineto traces.

    This metric measures how much communication overlaps with computation in time,
    indicating efficient scheduling.

    Args:
        directory (str): Path to the directory containing Kineto trace files.

    Returns:
        float: Overlap percentage (0-100).
    """

    trace_file = os.path.join(directory, "kineto_trace_0.json")

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Categorize events into communication and computation
        comm_events = []
        comp_events = []

        comm_kernels = ["nccl", "AllReduce", "AllGather", "ReduceScatter", "AllToAll"]
        comp_kernels = ["gemm", "conv", "matmul", "attention"]

        for event in trace_data.get("traceEvents", []):
            if event.get("cat") != "kernel":
                continue

            name = event.get("name", "").lower()
            ts = event.get("ts", 0)  # timestamp in microseconds
            dur = event.get("dur", 0)  # duration in microseconds

            if dur == 0:
                continue

            event_info = (ts, ts + dur)

            # Check if communication kernel
            if any(kw in name for kw in comm_kernels):
                comm_events.append(event_info)
            # Check if computation kernel
            elif any(kw in name for kw in comp_kernels):
                comp_events.append(event_info)

        if not comm_events or not comp_events:
            print("Warning: No communication or computation events found")
            return 0.0

        # Calculate overlap
        total_comm_time = sum(end - start for start, end in comm_events)
        overlap_time = 0.0

        for comm_start, comm_end in comm_events:
            for comp_start, comp_end in comp_events:
                # Calculate intersection
                overlap_start = max(comm_start, comp_start)
                overlap_end = min(comm_end, comp_end)

                if overlap_start < overlap_end:
                    overlap_time += (overlap_end - overlap_start)

        # Calculate overlap percentage
        if total_comm_time > 0:
            overlap_pct = (overlap_time / total_comm_time) * 100
        else:
            overlap_pct = 0.0

        return overlap_pct

    except FileNotFoundError:
        print(f"File not found: {trace_file}")
        return 0.0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")
        return 0.0
    except Exception as e:
        print(f"Error calculating overlap: {e}")
        return 0.0
