import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from tools.trace_analyzer import TraceAnalyzer
except ImportError:
    # Fallback
    from trace_analyzer import TraceAnalyzer

def metric_cal(directory: str) -> float:
    """
    Calculate Pipeline Bubble Ratio (%) from trace.
    Supports both Kineto JSON (kineto_trace_0.json) and Nsys CSV (cuda_gpu_trace.csv).
    """
    # Try nsys CSV first
    trace_path = os.path.join(directory, "cuda_gpu_trace.csv")
    if not os.path.exists(trace_path):
        # Fallback to Kineto
        trace_path = os.path.join(directory, "kineto_trace_0.json")
    
    if not os.path.exists(trace_path):
        print(f"No trace file found in {directory} (checked cuda_gpu_trace.csv and kineto_trace_0.json)")
        return 0.0
        
    try:
        analyzer = TraceAnalyzer(trace_path)
        return analyzer.calculate_bubble_ratio()
    except Exception as e:
        print(f"Error calculating Bubble Ratio: {e}")
        return 0.0
