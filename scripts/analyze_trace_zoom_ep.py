import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import glob
import gzip

def parse_traces(dir_path):
    """
    Parses all .pt.trace.json.gz files in the directory.
    Returns a dict: {rank_id: events_list}
    """
    print(f"Searching for trace files in {dir_path}...")
    files = glob.glob(os.path.join(dir_path, "*.pt.trace.json.gz"))
    
    if not files:
        print(f"No trace files found in {dir_path}")
        return {}
        
    # Filter files by modification time to group them (e.g. within 60s of the newest file)
    if files:
        latest_file = max(files, key=os.path.getmtime)
        latest_time = os.path.getmtime(latest_file)
        # Keep files modified within 5 minutes of the latest one
        files = [f for f in files if abs(os.path.getmtime(f) - latest_time) < 300]
        
    print(f"Found {len(files)} relevant trace files.")
    
    traces = {}
    for i, file_path in enumerate(sorted(files)):
        try:
            basename = os.path.basename(file_path)
            parts = basename.split('.')
            pid_part = parts[0].split('_')[1]
            rank_id = f"Rank {i} (PID {pid_part})"
        except:
            rank_id = f"Rank {i}"
            
        print(f"Loading {rank_id} from {file_path}...")
        try:
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'traceEvents' in data:
                    traces[rank_id] = data['traceEvents']
                else:
                    traces[rank_id] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return traces

def get_gpu_events(events):
    gpu_events = []
    for e in events:
        cat = e.get('cat', '')
        name = e.get('name', '').lower()
        
        if cat in ['kernel', 'gpu_op']:
            gpu_events.append(e)
        elif 'dur' in e:
            if cat == 'python_function':
                continue
            if 'nvtx' in cat or 'nvtx' in name:
                 gpu_events.append(e)
            elif 'nccl' in name or 'kernel' in name:
                 gpu_events.append(e)
            
    gpu_events.sort(key=lambda x: x.get('ts', 0))
    return gpu_events

def classify_event(event):
    name = event.get('name', '').lower()
    
    if 'tp:' in name: return 'TP: AllReduce'
    if 'ep:' in name: return 'EP: AllGather'
        
    if 'cross_device_reduce' in name: return 'TP: CustomReduce'
    if 'custom_all_reduce' in name: return 'TP: CustomAllReduce'
    
    if 'nccl' in name:
        if 'allreduce' in name: return 'NCCL: AllReduce'
        elif 'allgather' in name: return 'NCCL: AllGather'
        elif 'reducescatter' in name: return 'NCCL: ReduceScatter'
        elif 'alltoall' in name: return 'NCCL: AllToAll'
        else: return 'NCCL: Other'
    
    if 'comm' in name or 'send' in name or 'recv' in name:
         return 'Comm (Other)'

    return 'Compute'

def analyze_traces(dir_path, output_dir):
    traces = parse_traces(dir_path)
    if not traces:
        return

    # Process each rank
    rank_events = {}
    min_ts = float('inf')
    max_ts = 0
    
    all_ep_events = []

    for rank, events in traces.items():
        gpu_events = get_gpu_events(events)
        processed = []
        for e in gpu_events:
            cat = classify_event(e)
            processed.append({
                'name': e.get('name'),
                'ts': e['ts'],
                'dur': e['dur'],
                'category': cat
            })
            min_ts = min(min_ts, e['ts'])
            max_ts = max(max_ts, e['ts'] + e['dur'])
            
            if 'EP' in cat or 'AllGather' in cat:
                all_ep_events.append(e['ts'] + e['dur']) # Use end time
                
        rank_events[rank] = processed

    if min_ts == float('inf'):
        print("No GPU events found.")
        return

    if not all_ep_events:
        print("No EP events found to zoom in on.")
        return

    # Find the last EP event timestamp
    last_ep_ts = max(all_ep_events)
    print(f"Last EP event end time: {last_ep_ts}")
    
    # Define 5ms window around last EP event
    # Let's show [last_ep - 4ms, last_ep + 1ms]
    window_size_ms = 5
    end_ms_offset = 1
    
    target_end_ts = last_ep_ts + (end_ms_offset * 1000)
    target_start_ts = target_end_ts - (window_size_ms * 1000)
    
    # Ensure we don't go out of bounds
    target_start_ts = max(target_start_ts, min_ts)
    target_end_ts = min(target_end_ts, max_ts)
    
    print(f"Zoom window: {target_start_ts} to {target_end_ts} (Duration: {(target_end_ts - target_start_ts)/1000:.2f} ms)")
    
    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'Compute': '#d3d3d3', # LightGray
        'TP: AllReduce': '#1f77b4', # Blue
        'TP: CustomReduce': '#aec7e8', # Light Blue
        'TP: CustomAllReduce': '#17becf', # Cyan
        'EP: AllGather': '#2ca02c', # Green
        'NCCL: AllReduce': '#9467bd', # Purple
        'NCCL: AllGather': '#8c564b', # Brown
        'NCCL: ReduceScatter': '#e377c2', # Pink
        'NCCL: AllToAll': '#bcbd22', # Olive
        'NCCL: Other': '#7f7f7f', # Gray
        'Comm (Other)': '#ff7f0e', # Orange
    }
    
    sorted_ranks = sorted(rank_events.keys())
    
    fig, ax = plt.subplots(figsize=(20, 6 + len(sorted_ranks)))
    
    # Plot each rank
    for r_idx, rank in enumerate(sorted_ranks):
        events = rank_events[rank]
        
        xranges = defaultdict(list)
        for e in events:
            # Check overlap with zoom window
            e_start = e['ts']
            e_end = e['ts'] + e['dur']
            
            if e_end < target_start_ts or e_start > target_end_ts:
                continue
                
            # Normalize to window start = 0ms for plotting
            disp_start_ms = (max(e_start, target_start_ts) - target_start_ts) / 1000.0
            disp_end_ms = (min(e_end, target_end_ts) - target_start_ts) / 1000.0
            disp_dur_ms = disp_end_ms - disp_start_ms
            
            if disp_dur_ms > 0:
                xranges[e['category']].append((disp_start_ms, disp_dur_ms))
        
        for cat, ranges in xranges.items():
            ax.broken_barh(ranges, (r_idx - 0.4, 0.8), 
                          facecolors=colors.get(cat, 'gray'), edgecolors='none', alpha=0.9)
                          
    ax.set_yticks(range(len(sorted_ranks)))
    ax.set_yticklabels(sorted_ranks)
    ax.set_xlim(0, (target_end_ts - target_start_ts) / 1000.0)
    ax.set_xlabel('Time (ms) relative to window start')
    ax.set_title(f'GPU Pipeline Zoomed (Last EP Event - 4ms to +1ms)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Only show legend for categories present in the plot
    present_cats = set()
    for rank in sorted_ranks:
        for e in rank_events[rank]:
            if e['ts'] + e['dur'] >= target_start_ts and e['ts'] <= target_end_ts:
                present_cats.add(e['category'])
                
    patches = [mpatches.Patch(color=colors.get(cat, 'gray'), label=cat) for cat in sorted(present_cats)]
    ax.legend(handles=patches, loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'timeline_zoom_last_ep_5ms.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved zoomed plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", help="Directory containing trace files")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    analyze_traces(args.trace_dir, args.output_dir)
