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
        # Try to extract rank from filename or just use index
        # Filename format: nid001032_2192102.1765159717502925510.pt.trace.json.gz
        # We can use the PID (2192102) as a unique identifier
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
            # Ignore python functions
            if cat == 'python_function':
                continue
            # Nsys NVTX events
            if 'nvtx' in cat or 'nvtx' in name:
                 gpu_events.append(e)
            # Fallback for kernels in nsys
            elif 'nccl' in name or 'kernel' in name:
                 gpu_events.append(e)
            
    gpu_events.sort(key=lambda x: x.get('ts', 0))
    return gpu_events

def classify_event(event):
    name = event.get('name', '').lower()
    
    # Heuristics for vLLM
    if 'tp:' in name: return 'Comm (TP)'
    if 'ep:' in name: return 'Comm (EP)'
        
    # vLLM custom kernels
    if 'cross_device_reduce' in name: return 'Comm (TP)'
    if 'custom_all_reduce' in name: return 'Comm (TP)'
    
    if 'nccl' in name:
        if 'allreduce' in name: return 'Comm (TP)'
        elif 'allgather' in name: return 'Comm (EP)'
        elif 'reducescatter' in name: return 'Comm (EP)'
        elif 'alltoall' in name: return 'Comm (EP)'
        else: return 'Comm (Other)'
    
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
        rank_events[rank] = processed

    if min_ts == float('inf'):
        print("No GPU events found.")
        return

    print(f"Global time range: {min_ts} to {max_ts} (Duration: {(max_ts - min_ts)/1000:.2f} ms)")
    
    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into 600ms chunks
    chunk_size_ms = 600
    total_duration_ms = (max_ts - min_ts) / 1000.0
    num_chunks = int(np.ceil(total_duration_ms / chunk_size_ms))
    
    print(f"Generating {num_chunks} timeline plots (600ms each)...")
    
    colors = {
        'Compute': '#d3d3d3', # LightGray
        'Comm (TP)': '#1f77b4', # Blue
        'Comm (EP)': '#2ca02c', # Green
        'Comm (Other)': '#ff7f0e', # Orange
    }
    
    # Sort ranks for consistent plotting order
    sorted_ranks = sorted(rank_events.keys())
    
    for i in range(num_chunks):
        start_ms = i * chunk_size_ms
        end_ms = (i + 1) * chunk_size_ms
        
        fig, ax = plt.subplots(figsize=(20, 6 + len(sorted_ranks))) # Adjust height based on ranks
        
        # Plot each rank
        for r_idx, rank in enumerate(sorted_ranks):
            events = rank_events[rank]
            
            xranges = defaultdict(list)
            for e in events:
                e_start_ms = (e['ts'] - min_ts) / 1000.0
                e_dur_ms = e['dur'] / 1000.0
                e_end_ms = e_start_ms + e_dur_ms
                
                # Check overlap with current chunk
                if e_end_ms < start_ms or e_start_ms > end_ms:
                    continue
                    
                # Clip to chunk boundaries for display
                disp_start = max(e_start_ms, start_ms)
                disp_end = min(e_end_ms, end_ms)
                disp_dur = disp_end - disp_start
                
                if disp_dur > 0:
                    xranges[e['category']].append((disp_start, disp_dur))
            
            # Draw bars for this rank
            # Y-position: r_idx
            for cat, ranges in xranges.items():
                ax.broken_barh(ranges, (r_idx - 0.4, 0.8), 
                              facecolors=colors.get(cat, 'gray'), edgecolors='none', alpha=0.9)
                              
        ax.set_yticks(range(len(sorted_ranks)))
        ax.set_yticklabels(sorted_ranks)
        ax.set_xlim(start_ms, end_ms)
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'GPU Pipeline Timeline ({start_ms}-{end_ms} ms)')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Legend
        patches = [mpatches.Patch(color=colors[cat], label=cat) for cat in colors.keys()]
        ax.legend(handles=patches, loc='upper right')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'timeline_{i:02d}_{int(start_ms)}_{int(end_ms)}ms.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    print(f"Saved {num_chunks} plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", help="Directory containing trace files")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    analyze_traces(args.trace_dir, args.output_dir)
