import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

import glob
import gzip

def parse_trace(file_path):
    # Handle directory input: find the newest .json or .json.gz file
    if os.path.isdir(file_path):
        print(f"Input is a directory. Searching for trace files in {file_path}...")
        # Look for nsys exported json first
        files = glob.glob(os.path.join(file_path, "trace.json"))
        if not files:
            # Look for worker traces first (usually contain hostname/pid)
            # Pattern: *.pt.trace.json or *.pt.trace.json.gz
            files = glob.glob(os.path.join(file_path, "*.pt.trace.json*"))
        if not files:
            # Fallback to kineto_trace_*.json
            files = glob.glob(os.path.join(file_path, "kineto_trace_*.json"))
        
        if not files:
            print(f"No trace files found in {file_path}")
            return []
            
        # Pick the largest file (likely contains the most events)
        file_path = max(files, key=os.path.getsize)
        print(f"Selected trace file: {file_path}")

    print(f"Loading trace file: {file_path}...")
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading/decoding JSON from {file_path}: {e}")
        return []
    
    if isinstance(data, dict) and 'traceEvents' in data:
        events = data['traceEvents']
    else:
        events = data
    return events

def get_gpu_events(events):
    gpu_events = []
    for e in events:
        # Kineto/PyTorch Profiler usually uses cat="kernel" or "gpu_op"
        # Nsys JSON export might use different categories, often "cuda_api", "kernel", "nvtx"
        # We look for kernels and NVTX ranges
        cat = e.get('cat', '')
        name = e.get('name', '').lower()
        
        if cat in ['kernel', 'gpu_op']:
            gpu_events.append(e)
        elif 'dur' in e:
            # Ignore python functions
            if cat == 'python_function':
                continue
                
            # Nsys NVTX events
            if 'nvtx' in cat or 'nvtx' in name:
                 gpu_events.append(e)
            # Fallback for kernels in nsys
            elif 'nccl' in name or 'kernel' in name:
                 gpu_events.append(e)
            
    # Sort by start time
    gpu_events.sort(key=lambda x: x.get('ts', 0))
    return gpu_events

def classify_event(event):
    name = event.get('name', '').lower()
    
    # Heuristics for vLLM
    # Check NVTX markers first (e.g. "tp:all_reduce")
    if 'tp:' in name:
        return 'Comm (TP)', 'TP'
    if 'ep:' in name:
        return 'Comm (EP)', 'EP'
        
    # vLLM custom kernels
    if 'cross_device_reduce' in name:
        return 'Comm (TP)', 'TP'
    if 'custom_all_reduce' in name:
        return 'Comm (TP)', 'TP'
    
    if 'nccl' in name:
        if 'allreduce' in name:
            return 'Comm (TP)', 'TP'
        elif 'allgather' in name:
            return 'Comm (EP)', 'EP'
        elif 'reducescatter' in name:
            return 'Comm (EP)', 'EP'
        elif 'alltoall' in name:
            return 'Comm (EP)', 'EP'
        else:
            return 'Comm (Other)', 'Other'
    
    # Fallback: Check for generic communication keywords
    if 'comm' in name or 'send' in name or 'recv' in name:
         return 'Comm (Other)', 'Other'

    return 'Compute', 'Compute'

def extract_size(event):
    args = event.get('args', {})
    # Common keys for size in bytes
    for key in ['bytes', 'msg_size', 'size', 'count', 'numel']: 
        if key in args:
            try:
                return float(args[key])
            except (ValueError, TypeError):
                continue
    return 0.0

def analyze_trace(file_path, output_dir):
    events = parse_trace(file_path)
    gpu_events = get_gpu_events(events)
    
    print(f"Found {len(gpu_events)} GPU events.")
    
    if not gpu_events:
        print("No GPU events found in trace. Ensure CUDA Graphs are disabled (enforce_eager=True) and profiling is successful.")
        print("Also ensure you are NOT running 'nsys profile' wrapping the python script, as it conflicts with Kineto.")
        return

    # Normalize time to start at 0
    start_time = gpu_events[0]['ts']
    
    processed_events = []
    
    # Track gaps for "Wait"
    intervals = []
    for e in gpu_events:
        s = e['ts']
        d = e['dur']
        intervals.append((s, s + d))
    
    intervals.sort()
    merged_intervals = []
    if intervals:
        curr_start, curr_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start < curr_end: # Overlap
                curr_end = max(curr_end, next_end)
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_intervals.append((curr_start, curr_end))
    
    wait_events = []
    if merged_intervals:
        last_end = merged_intervals[0][1]
        for start, end in merged_intervals[1:]:
            gap = start - last_end
            if gap > 10: # Filter tiny gaps
                wait_events.append({
                    'name': 'Idle',
                    'ts': last_end,
                    'dur': gap,
                    'category': 'Wait',
                    'type': 'Wait',
                    'size': 0
                })
            last_end = end

    comm_events = []
    for e in gpu_events:
        cat, type_ = classify_event(e)
        processed_events.append({
            'name': e.get('name'),
            'ts': e['ts'],
            'dur': e['dur'],
            'category': cat,
            'type': type_,
            'size': extract_size(e)
        })
        if 'Comm' in cat:
            comm_events.append(processed_events[-1])

    print(f"Found {len(comm_events)} communication events.")

    all_plot_events = processed_events + wait_events
    all_plot_events.sort(key=lambda x: x['ts'])
    
    # --- Plotting ---
    os.makedirs(output_dir, exist_ok=True)
    
    plot_timeline(all_plot_events, start_time, output_dir)
    plot_bandwidth(comm_events, start_time, output_dir)

def plot_timeline(events, start_offset, output_dir):
    print("Generating Timeline Plot...")
    fig, ax = plt.subplots(figsize=(20, 8))
    
    colors = {
        'Compute': '#d3d3d3', # LightGray
        'Comm (TP)': '#1f77b4', # Blue
        'Comm (EP)': '#2ca02c', # Green
        'Comm (Other)': '#ff7f0e', # Orange
        'Wait': '#d62728' # Red
    }
    
    y_map = {
        'Compute': 0,
        'Comm (TP)': 1,
        'Comm (EP)': 2,
        'Comm (Other)': 3,
        'Wait': 4
    }
    
    xranges = defaultdict(list)
    
    for e in events:
        cat = e['category']
        start = (e['ts'] - start_offset) / 1000.0 # ms
        dur = e['dur'] / 1000.0 # ms
        xranges[cat].append((start, dur))

    for cat, ranges in xranges.items():
        y_pos = y_map.get(cat, 0)
        ax.broken_barh(ranges, (y_pos - 0.4, 0.8), 
                      facecolors=colors.get(cat, 'gray'), edgecolors='none', alpha=0.9)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel('Time (ms)')
    ax.set_title('GPU Pipeline Timeline')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    patches = [mpatches.Patch(color=colors[cat], label=cat) for cat in y_map.keys() if cat in xranges]
    ax.legend(handles=patches, loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'timeline.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved timeline to {out_path}")
    plt.close()

def plot_bandwidth(events, start_offset, output_dir):
    print("Generating Bandwidth Plot...")
    if not events:
        print("No communication events for bandwidth plot.")
        return

    times = []
    bws = []
    colors = []
    
    has_valid_size = False
    
    for e in events:
        if e['size'] > 0 and e['dur'] > 0:
            has_valid_size = True
            start = (e['ts'] - start_offset) / 1000.0 # ms
            dur = e['dur'] / 1000.0 # ms
            mid = start + dur/2
            
            # GB/s = (Bytes / 1e9) / (us / 1e6) = (Bytes / 1000) / us
            bw = (e['size'] / 1e9) / (e['dur'] / 1e6)
            
            times.append(mid)
            bws.append(bw)
            
            if 'TP' in e['category']:
                colors.append('#1f77b4') # Blue
            elif 'EP' in e['category']:
                colors.append('#2ca02c') # Green
            else:
                colors.append('#ff7f0e') # Orange

    if not has_valid_size:
        print("No events with size information found in trace. Cannot plot bandwidth.")
        return

    fig, ax = plt.subplots(figsize=(20, 6))
    
    scatter = ax.scatter(times, bws, c=colors, alpha=0.7, s=20)
    
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='TP Bandwidth'),
        mpatches.Patch(color='#2ca02c', label='EP Bandwidth'),
        mpatches.Patch(color='#ff7f0e', label='Other Bandwidth')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Communication Bandwidth over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'bandwidth.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved bandwidth plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GPU Trace Timeline")
    parser.add_argument("trace_file", help="Path to Kineto JSON trace file")
    parser.add_argument("--output-dir", default="trace_analysis", help="Output directory for plots")
    args = parser.parse_args()
    
    analyze_trace(args.trace_file, args.output_dir)
