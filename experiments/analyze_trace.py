import sqlite3
import pandas as pd
import bisect
import sys

db_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4-batch256.sqlite"

try:
    conn = sqlite3.connect(db_path)
except Exception as e:
    print(f"Error connecting to DB: {e}")
    sys.exit(1)

print("Loading StringIds...")
strings = pd.read_sql("SELECT id, value FROM StringIds", conn)
str_map = dict(zip(strings['id'], strings['value']))

print("Loading NVTX Events...")
nvtx_query = "SELECT start, end, globalTid, textId, text FROM NVTX_EVENTS"
nvtx = pd.read_sql(nvtx_query, conn)
nvtx['mapped_text'] = nvtx['textId'].map(str_map)
nvtx['final_text'] = nvtx['text'].fillna(nvtx['mapped_text'])

# Filter for EP ranges.
ep_ranges = nvtx[nvtx['final_text'].str.contains("EP_", na=False)].copy()
print(f"Found {len(ep_ranges)} EP NVTX ranges.")

print("Loading Kernels...")
kernel_query = """
    SELECT k.start, k.end, k.deviceId, k.correlationId, k.demangledName as nameId
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
"""
kernels = pd.read_sql(kernel_query, conn)
kernels['name'] = kernels['nameId'].map(str_map)

# Filter for cross_device_reduce kernels
comm_kernels = kernels[kernels['name'].str.contains("cross_device_reduce", case=False, na=False)].copy()
print(f"Found {len(comm_kernels)} cross_device_reduce kernels.")

print("Loading Runtime Events...")
runtime_query = "SELECT correlationId, start as cpu_start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
runtime = pd.read_sql(runtime_query, conn)

# Join Kernels with Runtime
comm_kernels = comm_kernels.merge(runtime, on='correlationId', how='left')
comm_kernels.dropna(subset=['globalTid'], inplace=True)

print("Classifying Kernels...")
# Pre-group ranges by TID for faster lookup
# Store as (start, end) tuples, sorted by start
ep_ranges_by_tid = {}
for tid, group in ep_ranges.groupby('globalTid'):
    # Sort by start
    sorted_group = group.sort_values('start')
    starts = sorted_group['start'].tolist()
    ends = sorted_group['end'].tolist()
    ep_ranges_by_tid[tid] = (starts, ends)

def classify_kernel(row):
    tid = row['globalTid']
    start = row['cpu_start']
    
    if tid not in ep_ranges_by_tid:
        return 'TP'
        
    starts, ends = ep_ranges_by_tid[tid]
    
    # Find the first range that starts after 'start'
    # The candidate range is the one before that index
    idx = bisect.bisect_right(starts, start)
    
    if idx > 0:
        # Check the range at idx-1
        # range is [starts[idx-1], ends[idx-1]]
        if start <= ends[idx-1]:
            return 'EP'
            
    return 'TP'

comm_kernels['type'] = comm_kernels.apply(classify_kernel, axis=1)

# Calculate Metrics
comm_kernels['duration_ns'] = comm_kernels['end'] - comm_kernels['start']
comm_kernels['duration_ms'] = comm_kernels['duration_ns'] / 1e6

# Group by Device and Type
stats = comm_kernels.groupby(['deviceId', 'type'])['duration_ms'].agg(['sum', 'count', 'mean'])

# Calculate Total Trace Duration per Device
device_stats = kernels.groupby('deviceId').agg({'start': 'min', 'end': 'max'})
device_stats['total_duration_ms'] = (device_stats['end'] - device_stats['start']) / 1e6

print("\n--- Analysis Results ---")
for device_id in sorted(stats.index.levels[0]):
    print(f"\nDevice {device_id}:")
    if device_id not in device_stats.index:
        print("  No kernel activity found.")
        continue
        
    total_dur = device_stats.loc[device_id, 'total_duration_ms']
    print(f"  Total Trace Duration: {total_dur:.2f} ms")
    
    if device_id in stats.index:
        dev_data = stats.loc[device_id]
        for comm_type in ['TP', 'EP']:
            if comm_type in dev_data.index:
                s = dev_data.loc[comm_type]
                overhead_pct = (s['sum'] / total_dur) * 100
                print(f"  {comm_type} Communication:")
                print(f"    Count: {int(s['count'])}")
                print(f"    Total Duration: {s['sum']:.2f} ms")
                print(f"    Avg Duration: {s['mean']:.2f} ms")
                print(f"    Overhead: {overhead_pct:.2f}%")
            else:
                print(f"  {comm_type} Communication: None")

conn.close()
