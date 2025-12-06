import os
import sys

# Allow running both as a package and a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_analyzer import TraceAnalyzer


def metric_cal(directory: str) -> float:
    """
    Communication overhead (%) computed from kineto_trace_0.json.
    Focuses on TP-related collectives (all-reduce/all-gather/reduce-scatter).
    """
    trace_path = os.path.join(directory, "kineto_trace_0.json")
    if not os.path.exists(trace_path):
        print(f"Kineto trace not found: {trace_path}")
        return 0.0

    try:
        analyzer = TraceAnalyzer(trace_path)
        return analyzer.calculate_comm_overhead()
    except Exception as exc:
        print(f"Error calculating comm overhead: {exc}")
        return 0.0
