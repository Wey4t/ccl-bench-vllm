import json
import os

def metric_cal(directory: str) -> float:
    """
    Calculate Average Time To First Token (TTFT) in ms.
    """
    timing_stats_path = os.path.join(directory, "timing_stats_0.json")
    
    try:
        with open(timing_stats_path, 'r') as f:
            timing_stats = json.load(f)
            
        # Return in ms
        return timing_stats.get('ttft_avg', 0.0) * 1000.0
        
    except Exception as e:
        print(f"Error calculating TTFT: {e}")
        return 0.0
