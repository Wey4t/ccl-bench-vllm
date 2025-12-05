import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from tools.trace_analyzer import TraceAnalyzer
except ImportError:
    # Fallback
    from trace_analyzer import TraceAnalyzer

def metric_cal(path: str) -> float:
    """
    Calculate Pipeline Bubble Ratio (%) from trace.
    Supports both Kineto JSON (kineto_trace_0.json) and Nsys CSV (cuda_gpu_trace.csv).
    Args:
        path: Directory containing trace files OR path to a specific trace file.
    """
    trace_path = path
    
    # If directory, look for default files
    if os.path.isdir(path):
        # Try nsys CSV first
        trace_path = os.path.join(path, "cuda_gpu_trace.csv")
        if not os.path.exists(trace_path):
            # Fallback to Kineto
            trace_path = os.path.join(path, "kineto_trace_0.json")
    
    if not os.path.exists(trace_path):
        print(f"No trace file found in or at {path}")
        return 0.0
        
    try:
        analyzer = TraceAnalyzer(trace_path)
        return analyzer.calculate_bubble_ratio()
    except Exception as e:
        print(f"Error calculating Bubble Ratio: {e}")
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_bubble_ratio.py <trace_directory_or_file>")
        sys.exit(1)
        
    path = sys.argv[1]
    ratio = metric_cal(path)
    print(f"Pipeline Bubble Ratio: {ratio:.2f}%")
