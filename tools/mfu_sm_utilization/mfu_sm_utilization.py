import json
import os
from typing import Dict


def metric_cal(directory: str) -> Dict[str, float]:
    """
    Calculate Model FLOPs Utilization (MFU) and Streaming Multiprocessor (SM) utilization.

    MFU measures the efficiency of the model's computation relative to theoretical peak.
    SM utilization indicates how well GPU streaming multiprocessors are being used.

    Args:
        directory (str): Path to the directory containing Kineto trace files.

    Returns:
        dict: Dictionary with 'mfu' and 'sm_utilization' keys, both as percentages (0-100).
    """

    trace_file = os.path.join(directory, "kineto_trace_0.json")

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # GPU theoretical specs (A100 example - should be read from workload card)
        # A100 has 312 TFLOPS (FP16/BF16) and 108 SMs
        theoretical_tflops = 312.0
        total_sms = 108

        # Computation kernels
        comp_patterns = ["gemm", "matmul", "conv", "attention", "elementwise"]

        total_flops = 0
        total_compute_time_sec = 0
        sm_utilization_samples = []

        for event in trace_data.get("traceEvents", []):
            if event.get("cat") != "kernel":
                continue

            name = event.get("name", "").lower()
            dur = event.get("dur", 0)  # microseconds

            if dur == 0:
                continue

            # Check if computation kernel
            if any(pattern in name for pattern in comp_patterns):
                args = event.get("args", {})

                # Extract FLOPs if available
                flops = args.get("flops", 0)
                if flops > 0:
                    total_flops += flops
                    total_compute_time_sec += dur / 1_000_000

                # Extract SM utilization if available
                sm_occupancy = args.get("est. achieved occupancy %", 0)
                if sm_occupancy > 0:
                    sm_utilization_samples.append(sm_occupancy)

        # Calculate MFU
        if total_compute_time_sec > 0 and total_flops > 0:
            achieved_tflops = (total_flops / total_compute_time_sec) / 1e12
            mfu = (achieved_tflops / theoretical_tflops) * 100
        else:
            print("Warning: Insufficient data to calculate MFU")
            mfu = 0.0

        # Calculate average SM utilization
        if sm_utilization_samples:
            sm_utilization = sum(sm_utilization_samples) / len(sm_utilization_samples)
        else:
            # Fallback: estimate from active kernel time
            kernel_events = []
            for event in trace_data.get("traceEvents", []):
                if event.get("cat") == "kernel":
                    dur = event.get("dur", 0)
                    if dur > 0:
                        kernel_events.append(dur)

            if kernel_events:
                # Rough estimate: assume SM utilization scales with kernel activity
                sm_utilization = min(100.0, (len(kernel_events) / total_sms) * 50)
            else:
                print("Warning: No SM utilization data available")
                sm_utilization = 0.0

        return {
            "mfu": mfu,
            "sm_utilization": sm_utilization
        }

    except FileNotFoundError:
        print(f"File not found: {trace_file}")
        return {"mfu": 0.0, "sm_utilization": 0.0}
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")
        return {"mfu": 0.0, "sm_utilization": 0.0}
    except Exception as e:
        print(f"Error calculating MFU/SM utilization: {e}")
        return {"mfu": 0.0, "sm_utilization": 0.0}
