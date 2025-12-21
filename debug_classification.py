import gzip
import json
import sys

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

file_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/deepseek-v2-lite-vllm-perlmutter-E3.3/nid001032_2192102.1765159717502925510.pt.trace.json.gz"

print(f"Analyzing {file_path}...")
try:
    with gzip.open(file_path, 'rt') as f:
        data = json.load(f)
        events = data.get('traceEvents', [])
        
        # Filter for GPU events exactly like the script
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
                     
    # Sort by start time
        
        from collections import Counter
        categories = Counter()
        other_names = Counter()
        
        for e in gpu_events:
            cat, type_ = classify_event(e)
            categories[cat] += 1
            if cat == 'Comm (Other)':
                other_names[e['name']] += 1
                
        print("\nCategories:")
        for cat, count in categories.items():
            print(f"{cat}: {count}")
            
        print("\nTop 'Comm (Other)' names:")
        for name, count in other_names.most_common(20):
            print(f"{count}: {name}")

except Exception as e:
    print(f"Error: {e}")
