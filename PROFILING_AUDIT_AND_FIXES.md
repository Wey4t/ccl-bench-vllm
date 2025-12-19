# Profiling Audit and Fixes (Perlmutter vLLM)

## Executive summary (what was wrong)
- Kineto/torch profiler was only attached to the driver process; resulting GPU traces contained almost no CUDA kernels, so comm_overhead/coll_call_num/bubble_ratio silently returned zeros.
- `vllm_profiler.py` only looked at the last iteration’s outputs and used incorrect percentile math, and iteration timings were taken without GPU synchronization (under-reported iteration time).
- Metric tools returned 0 on missing/invalid traces and `nsys` capture was disabled via `--capture-range=cudaProfilerApi` without any start/stop calls, producing empty reports.

## Detailed fix log (file → change → why)
- `vllm_profiler.py`: aggregate outputs across profiled iterations; synchronize CUDA before timing; fixed percentile calculation; tolerate missing metrics with `None` instead of fake zeros. Ensures TTFT/TPOT/iteration time are computed from all profiled steps with correct units.
- `tools/trace_analyzer.py`: normalize kernel detection, require kernel events, and raise on empty/malformed traces; added NCCL keyword; improved duration validation. Prevents “0%” metrics when CUDA capture is missing.
- `tools/comm_overhead/comm_overhead.py`: support multiple ranks and nsys-exported `cuda_gpu_trace*.csv`; fail if no trace. Avoids false zeros and lets comm overhead use nsys data.
- `tools/comm_comp_overlap/comm_comp_overlap.py`: support multiple traces, require comm/compute kernels, fail loudly if absent. Removes heuristic fallbacks.
- `tools/coll_call_num/coll_call_num.py`: parse all kineto/nsys traces via `TraceAnalyzer`, count NCCL kernels, and fail if traces are absent. Prevents silent zeros.
- `tools/pipeline_bubble_ratio/pipeline_bubble_ratio.py`: average across kineto or nsys traces; fail if missing. Avoids bogus “0” bubbles when traces are absent.
- `tools/throughput_tokens_sec/throughput_tokens_sec.py`, `iteration_wall_clock.py`, `ttft.py`, `tpot.py`: add timing_stats rank fallback, validate fields, and raise on bad inputs. Stops silent 0.0 outputs from missing files.
- `experiments/analyze_results.py`: use timing_stats fallback, allow workload_card override, drop NaNs in plots, and store `None` for unmeasured trace metrics instead of zeros. Prevents misleading aggregates.
- `experiments/slurm_scripts/run_with_nsys.sh`: remove capture-range gating, ensure trace dir exists, export `cuda_gpu_trace.csv` via `nsys stats`, and detect report filename. Produces usable nsys traces for tools.
- `README.md`: point to this audit doc.

## Truth status table
| metric | grounded source | status | notes |
| --- | --- | --- | --- |
| avg_iteration_time | timing_stats_0/rank0.json | MEASURED | From profiled iterations only; warmup excluded. |
| ttft_ms | timing_stats_0/rank0.json | MEASURED | Derived from vLLM outputs/stat logger; `None` if absent. |
| tpot_ms | timing_stats_0/rank0.json | MEASURED | Same as above; `None` if absent. |
| steps_per_sec | timing_stats_0/rank0.json | MEASURED | 1 / avg_iteration_time; tokens per step unknown. |
| throughput_tokens_sec | timing_stats + workload_card/config | MEASURED | Assumes batch_size * seq_len tokens per iter; fails if metadata missing. |
| comm_overhead_pct | kineto_trace_* or cuda_gpu_trace*.csv | MEASURED when kernel events present | Raises if no kernel events (e.g., missing CUDA capture). |
| bubble_ratio_pct | kineto_trace_* or cuda_gpu_trace*.csv | MEASURED when compute kernels present | Raises if no compute kernels. |
| sm_efficiency_pct | kineto_trace_* or cuda_gpu_trace*.csv | MEASURED when kernel events present | Raises if no kernel events. |
| coll_call_num | kineto_trace_* or cuda_gpu_trace*.csv | MEASURED when NCCL kernels present | Counts NCCL kernels across traces. |
| comm_comp_overlap | kineto_trace_* or cuda_gpu_trace*.csv | MEASURED when both comm/compute kernels present | Heuristic overlap; fails if either category missing. |
| MFU / detailed SM metrics | n/a | NOT MEASURED | Requires Nsight Compute (ncu) or model FLOP accounting; not implemented. |

## How to run profiling on Perlmutter
1) Generate configs  
```bash
python experiments/generate_all_configs.py
```

2) Run TP=1/2/4 (Kineto + torch ET)  
```bash
sbatch experiments/slurm_scripts/run_E1.1.sh   # TP=1
sbatch experiments/slurm_scripts/run_E1.2.sh   # TP=2
sbatch experiments/slurm_scripts/run_E1.3.sh   # TP=4
```

3) Run TP=1/2/4 with Nsys enabled (produces nsys_report.*rep + cuda_gpu_trace.csv)  
```bash
sbatch --export=CONFIG=E1.1_llama8b_baseline.yaml experiments/slurm_scripts/run_with_nsys.sh
sbatch --export=CONFIG=E1.2_llama8b_tp2.yaml     experiments/slurm_scripts/run_with_nsys.sh
sbatch --export=CONFIG=E1.3_llama8b_tp4.yaml     experiments/slurm_scripts/run_with_nsys.sh
```

4) Compute metrics from traces  
- Timing-derived metrics (ttft_ms, tpot_ms, iteration wall clock):  
  ```bash
  python tools/main.py --trace trace_collection/llama-8b-tp1 --metric ttft
  python tools/main.py --trace trace_collection/llama-8b-tp1 --metric tpot
  python tools/main.py --trace trace_collection/llama-8b-tp1 --metric iteration_wall_clock
  python tools/main.py --trace trace_collection/llama-8b-tp1 --metric throughput_tokens_sec
  ```
- Trace-derived metrics (require kineto_trace_* or cuda_gpu_trace*.csv with kernel events):  
  ```bash
  python tools/main.py --trace trace_collection/llama-8b-tp2 --metric comm_overhead
  python tools/main.py --trace trace_collection/llama-8b-tp2 --metric coll_call_num
  python tools/main.py --trace trace_collection/llama-8b-tp2 --metric comm_comp_overlap
  python tools/main.py --trace trace_collection/llama-8b-tp2 --metric iteration_wall_clock
  ```
- After an Nsys run, you can re-export CUDA traces or get top kernels:  
  ```bash
  NSYS_REP=$(ls trace_collection/llama-8b-tp4/nsys_report.*rep | head -n1)
  nsys stats --report cuda_gpu_trace --format csv \
    -o trace_collection/llama-8b-tp4/cuda_gpu_trace "$NSYS_REP"
  nsys stats --report gpukernsum --format csv \
    -o trace_collection/llama-8b-tp4/gpukernsum "$NSYS_REP"
  ```

5) Aggregate results across experiments  
```bash
python experiments/analyze_results.py
```

## Known limitations / next steps
- Kineto traces still come from the driver process; for multi-GPU tensor parallel runs, CUDA kernels may live in worker processes. Nsys captures are recommended for reliable GPU-side metrics until per-rank Kineto capture is added.
- MFU/accurate SM efficiency and hardware counters require Nsight Compute (ncu) and model FLOP accounting; currently NOT MEASURED.
- Overlap/bubble computations remain keyword-based heuristics on kernel names; rely on nsys or future kernel classification for higher fidelity.
- timing_stats only cover profiled iterations; include warmup handling if longer steady-state is needed.

## Minimal sanity checks
- Verified tools now raise on missing traces/fields instead of returning 0 (e.g., removing kineto files causes comm_overhead to raise).
- Confirmed `throughput_tokens_sec` and iteration wall clock pick up timing_stats_rank0.json as well as timing_stats_0.json.
- Ensured `run_with_nsys.sh` emits `cuda_gpu_trace.csv` via `nsys stats`, enabling TraceAnalyzer to process nsys runs.
