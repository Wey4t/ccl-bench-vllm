import json
import sys

def inspect_trace(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    events = data.get('traceEvents', [])
    print(f"Total events: {len(events)}")
    
    kernel_events = [e for e in events if e.get('cat') in ['kernel', 'gpu_op']]
    print(f"Kernel events: {len(kernel_events)}")
    
    if kernel_events:
        print("First 10 kernel names:")
        for e in kernel_events[:10]:
            print(f"  {e.get('name')}")
            
    # Check for any event with 'nccl' in name
    nccl_events = [e for e in events if 'nccl' in e.get('name', '').lower()]
    print(f"NCCL events: {len(nccl_events)}")

    # Check for any event with 'tp:' in name
    tp_events = [e for e in events if 'tp:' in e.get('name', '').lower()]
    print(f"TP events: {len(tp_events)}")

if __name__ == "__main__":
    inspect_trace(sys.argv[1])
