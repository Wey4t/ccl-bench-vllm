import json
import gzip
import sys
import os

def check_nccl_sequence(file_path):
    print(f"Checking {file_path}...")
    try:
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    events = data.get('traceEvents', [])
    if not events:
        events = data 

    # Filter for NCCL events
    nccl_events = []
    for e in events:
        name = e.get('name', '').lower()
        cat = e.get('cat', '').lower()
        
        if 'nccl' in name:
            nccl_events.append(e)
    
    nccl_events.sort(key=lambda x: x.get('ts', 0))

    print(f"Found {len(nccl_events)} NCCL events.")
    
    count = 0
    for i, e in enumerate(nccl_events):
        name = e.get('name', '')
        cat = e.get('cat', '')
        ts = e.get('ts', 0)
        dur = e.get('dur', 0)
        
        # Only print kernels or interesting ops
        if 'kernel' in cat or 'gpu' in cat or 'nccl' in name:
             print(f"{i}: [{cat}] {name} at {ts} (dur: {dur})")
             count += 1
             if count > 50:
                 print("... (truncating)")
                 break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_nccl_sequence.py <trace_file>")
        sys.exit(1)
    
    check_nccl_sequence(sys.argv[1])
