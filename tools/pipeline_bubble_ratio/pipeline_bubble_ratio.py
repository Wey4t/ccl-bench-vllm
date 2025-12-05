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
    Calculate Pipeline Bubble Ratio (%) from Kineto trace.
    """
    kineto_trace_path = os.path.join(directory, "kineto_trace_0.json")
    
    if not os.path.exists(kineto_trace_path):
        return 0.0
        
    try:
        analyzer = TraceAnalyzer(kineto_trace_path)
        return analyzer.calculate_bubble_ratio()
    except Exception as e:
        print(f"Error calculating Bubble Ratio: {e}")
        return 0.0
