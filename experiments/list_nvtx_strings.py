import sqlite3
import pandas as pd

db_path = "/pscratch/sd/n/nw388/final/ccl-bench-vllm/trace_collection/nsys-4/ep4_wosp.sqlite"
conn = sqlite3.connect(db_path)

strings = pd.read_sql("SELECT id, value FROM StringIds", conn)
str_map = dict(zip(strings['id'], strings['value']))

nvtx = pd.read_sql("SELECT text, textId FROM NVTX_EVENTS", conn)
# Map textId
nvtx['mapped_text'] = nvtx['textId'].map(str_map)
# Combine
nvtx['final_text'] = nvtx['text'].fillna(nvtx['mapped_text'])

print(nvtx['final_text'].unique().tolist())

conn.close()
