import sqlite3

db_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4_wosp.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def print_schema(table):
    cursor.execute(f"PRAGMA table_info({table})")
    print(f"--- {table} ---")
    for col in cursor.fetchall():
        print(col)

print_schema("NVTX_EVENTS")
print_schema("CUPTI_ACTIVITY_KIND_KERNEL")
print_schema("CUPTI_ACTIVITY_KIND_RUNTIME")

conn.close()
