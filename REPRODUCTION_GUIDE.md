# Task B 实验复现指南 (Reproduction Guide)

本指南将指导你如何复现 Task B (Scaling Efficiency & Pipeline Bubbles) 的实验，并获取所有 6 个关键性能指标。

## 0. 环境准备
确保你已经连接到 NERSC Perlmutter，并且位于项目根目录：
```bash
cd ~/CS5470/project/ccl-bench-vllm
git pull origin task-b-impl
```

## 1. 运行实验 (Data Collection)
我们需要对 **E2.1**, **E2.2**, **E2.3** 三个配置分别进行 Profiling。

### 实验 1: E2.1 (TP=4, PP=1)
*   **目标**: 获取 Baseline 延迟和通信开销。
*   **命令**:
    ```bash
    # 1. 运行实验并抓取 Trace
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --trace-fork-before-exec=true \
      --delay=45 \
      --output=$SCRATCH/E2.1_qwen_tp4_trace \
      --force-overwrite=true \
      python vllm_profiler.py --config experiments/configs/E2.1_qwen_tp4.yaml

    # 2. 导出 CSV 数据
    nsys stats --force-export=true --format csv --report cuda_gpu_trace \
      --output $SCRATCH/E2.1_cuda_gpu_trace \
      $SCRATCH/E2.1_qwen_tp4_trace.nsys-rep
    ```
*   **记录数据**: 
    *   在终端输出中找到 `Average TTFT` 和 `Average TPOT`，记下来。

### 实验 2: E2.2 (TP=2, PP=2)
*   **目标**: 测量 2-Stage Pipeline 的气泡。
*   **命令**:
    ```bash
    # 1. 运行实验
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --trace-fork-before-exec=true \
      --delay=45 \
      --output=$SCRATCH/E2.2_qwen_tp2_pp2_trace \
      --force-overwrite=true \
      python vllm_profiler.py --config experiments/configs/E2.2_qwen_tp2_pp2.yaml

    # 2. 导出 CSV
    nsys stats --force-export=true --format csv --report cuda_gpu_trace \
      --output $SCRATCH/E2.2_cuda_gpu_trace \
      $SCRATCH/E2.2_qwen_tp2_pp2_trace.nsys-rep
    ```
*   **记录数据**: 终端中的 TTFT 和 TPOT。

### 实验 3: E2.3 (TP=1, PP=4)
*   **目标**: 测量 4-Stage Pipeline 的气泡。
*   **命令**:
    ```bash
    # 1. 运行实验
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --trace-fork-before-exec=true \
      --delay=45 \
      --output=$SCRATCH/E2.3_qwen_pp4_trace \
      --force-overwrite=true \
      python vllm_profiler.py --config experiments/configs/E2.3_qwen_pp4.yaml

    # 2. 导出 CSV
    nsys stats --force-export=true --format csv --report cuda_gpu_trace \
      --output $SCRATCH/E2.3_cuda_gpu_trace \
      $SCRATCH/E2.3_qwen_pp4_trace.nsys-rep
    ```
*   **记录数据**: 终端中的 TTFT 和 TPOT。

---

## 2. 计算指标 (Metric Calculation)
我们提供了一个脚本可以自动从 CSV 中计算 **Bubble Ratio**, **Communication Overhead**, 和 **SM Efficiency**。

运行以下命令：

```bash
# 计算 E2.1 指标 (注意：E2.1 的 Bubble Ratio 无效，应记为 0)
python tools/calc_all_metrics.py $SCRATCH/E2.1_cuda_gpu_trace_cuda_gpu_trace.csv

# 计算 E2.2 指标
python tools/calc_all_metrics.py $SCRATCH/E2.2_cuda_gpu_trace_cuda_gpu_trace.csv

# 计算 E2.3 指标
python tools/calc_all_metrics.py $SCRATCH/E2.3_cuda_gpu_trace_cuda_gpu_trace.csv
```

---

## 3. 生成汇总报表 (Summary & Plotting)
收集好所有数据（TTFT, TPOT 以及上面算出的 3 个指标）后，你可以更新汇总脚本并生成图表。

1.  **编辑数据**:
    打开 `experiments/generate_task_b_summary.py`，修改 `data` 列表中的数值为你刚刚测得的实际数据：
    ```python
    data = [
        {
            "experiment": "E2.1_qwen_tp4",
            "ttft_ms": YOUR_VALUE, 
            "tpot_ms": YOUR_VALUE,
            ...
        },
        ...
    ]
    ```

2.  **生成图表**:
    ```bash
    python experiments/generate_task_b_summary.py
    ```

3.  **结果文件**:
    脚本运行后，会在 `experiments/` 目录下生成：
    *   `results_summary.csv`: 包含所有数据的汇总表。
    *   `plot_ttft.png`, `plot_bubble_ratio.png` 等 6 张性能对比图。

4.  **下载到本地**:
    在本地终端运行：
    ```powershell
    scp your_user@perlmutter.nersc.gov:~/CS5470/project/ccl-bench-vllm/experiments/plot_*.png .
    ```
