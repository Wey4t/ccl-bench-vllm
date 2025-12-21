import gzip
import json
import sys

file_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/nid001100_1843139.1765159501397732601.pt.trace.json.gz"

print(f"Reading {file_path}...")
try:
    with gzip.open(file_path, 'rt') as f:
        data = json.load(f)
        events = data.get('traceEvents', [])
        print(f"Total events: {len(events)}")
        
        # Check for other communication kernels
        comm_keywords = ['comm', 'send', 'recv']
        comm_events = [e for e in events if e.get('cat') in ['kernel', 'gpu_op'] and any(k in e.get('name', '').lower() for k in comm_keywords) and 'nccl' not in e.get('name', '').lower()]
        
        print(f"\nFound {len(comm_events)} non-NCCL communication kernels.")
        from collections import Counter
        counts = Counter([e['name'] for e in comm_events])
        for name, count in counts.most_common(20):
            print(f"{count}: {name}")
            
except Exception as e:
    print(f"Error: {e}")
