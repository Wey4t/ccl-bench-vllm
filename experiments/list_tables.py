import sqlite3

db_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4_wosp.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(table[0])

conn.close()
