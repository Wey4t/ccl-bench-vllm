import json
import os
from typing import List, Tuple


def metric_cal(directory: str) -> float:
    """
    Calculate pipeline bubble size (idle time) as a percentage.

    This metric measures the fraction of time spent idle (bubbles) in pipeline
    parallel execution, indicating pipeline efficiency.

    Args:
        directory (str): Path to the directory containing Kineto trace files.

    Returns:
        float: Bubble size as percentage of total execution time (0-100).
    """

    trace_file = os.path.join(directory, "kineto_trace_0.json")

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Find all kernel execution events
        kernel_events = []
        for event in trace_data.get("traceEvents", []):
            if event.get("cat") == "kernel":
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                if dur > 0:
                    kernel_events.append((ts, ts + dur))

        if not kernel_events:
            print("Warning: No kernel events found")
            return 0.0

        # Sort events by start time
        kernel_events.sort()

        # Find the overall time span
        min_time = kernel_events[0][0]
        max_time = max(end for _, end in kernel_events)
        total_time = max_time - min_time

        if total_time == 0:
            return 0.0

        # Calculate active time by merging overlapping intervals
        merged_intervals = []
        current_start, current_end = kernel_events[0]

        for start, end in kernel_events[1:]:
            if start <= current_end:
                # Overlapping or adjacent, merge
                current_end = max(current_end, end)
            else:
                # Gap found, save current interval
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end

        # Don't forget the last interval
        merged_intervals.append((current_start, current_end))

        # Calculate total active time
        active_time = sum(end - start for start, end in merged_intervals)

        # Calculate bubble (idle) time
        idle_time = total_time - active_time
        bubble_percentage = (idle_time / total_time) * 100

        return bubble_percentage

    except FileNotFoundError:
        print(f"File not found: {trace_file}")
        return 0.0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")
        return 0.0
    except Exception as e:
        print(f"Error calculating pipeline bubble: {e}")
        return 0.0
