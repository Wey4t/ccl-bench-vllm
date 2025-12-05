import json
from typing import Dict, List, Any

class TraceAnalyzer:
    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        self.events = self._load_trace()
        
    def _load_trace(self) -> List[Dict[str, Any]]:
        """Load and parse the Kineto trace JSON file."""
        try:
            with open(self.trace_path, 'r') as f:
                data = json.load(f)
                events = []
                if isinstance(data, dict) and 'traceEvents' in data:
                    events = data['traceEvents']
                elif isinstance(data, list):
                    events = data
                else:
                    print(f"Warning: Unexpected trace format in {self.trace_path}")
                    return []
                
                print(f"[DEBUG] Loaded trace {self.trace_path} with {len(events)} events")
                nccl_events = [e for e in events if 'nccl' in e.get('name', '').lower()]
                print(f"[DEBUG] Found {len(nccl_events)} NCCL events")
                return events
        except Exception as e:
            print(f"Error loading trace {self.trace_path}: {e}")
            return []

    def calculate_comm_overhead(self) -> float:
        """
        Calculate Communication Overhead (%).
        Sum of duration of NCCL kernels / Total trace duration.
        """
        if not self.events:
            return 0.0
            
        comm_time = 0.0
        min_ts = float('inf')
        max_ts = float('-inf')
        
        for event in self.events:
            if 'ts' not in event or 'dur' not in event:
                continue
                
            ts = event['ts']
            dur = event['dur']
            name = event.get('name', '').lower()
            
            min_ts = min(min_ts, ts)
            max_ts = max(max_ts, ts + dur)
            
            if 'nccl' in name:
                comm_time += dur
                
        total_duration = max_ts - min_ts
        if total_duration <= 0:
            return 0.0
            
        return (comm_time / total_duration) * 100.0

    def calculate_bubble_ratio(self) -> float:
        """
        Estimate Pipeline Bubble Ratio (%).
        """
        if not self.events:
            return 0.0
            
        compute_events = []
        for event in self.events:
            if event.get('cat') == 'kernel' and 'nccl' not in event.get('name', '').lower():
                if 'ts' in event and 'dur' in event:
                    compute_events.append((event['ts'], event['ts'] + event['dur']))
        
        if not compute_events:
            return 0.0
            
        compute_events.sort(key=lambda x: x[0])
        
        merged = []
        if compute_events:
            curr_start, curr_end = compute_events[0]
            for next_start, next_end in compute_events[1:]:
                if next_start < curr_end:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))
            
        active_time = sum(end - start for start, end in merged)
        
        total_duration = compute_events[-1][1] - compute_events[0][0]
        
        if total_duration <= 0:
            return 0.0
            
        idle_time = total_duration - active_time
        return (idle_time / total_duration) * 100.0

    def calculate_sm_efficiency(self) -> float:
        return 0.0
