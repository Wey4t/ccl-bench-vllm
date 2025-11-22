# vLLM Parallelism Profiling Experiments

This directory contains infrastructure for profiling vLLM inference with different parallelism strategies on Perlmutter.

## Directory Structure

```
experiments/
├── README.md                      # This file
├── configs/                       # Experiment configurations (YAML)
├── slurm_scripts/                 # SLURM job scripts for Perlmutter
├── generate_all_configs.py        # Generate all experiment configs
├── generate_workload_card.py      # Generate workload cards
└── analyze_results.py             # Analyze and visualize results
```

## Experiments Overview

### E1: Llama-8B Single-Parallelism
- **E1.1**: Baseline (1 GPU, TP=1)
- **E1.2**: Tensor Parallelism (2 GPUs, TP=2)
- **E1.3**: Tensor Parallelism (4 GPUs, TP=4)
- **E1.4**: Data Parallelism (2 GPUs, DP=2)
- **E1.6**: Pipeline Parallelism (2 GPUs, PP=2)

### E2: Multi-Parallelism
- **E2.1**: Llama-8B (4 GPUs, TP=2, DP=2)
- **E2.2**: Llama-8B (4 GPUs, TP=2, PP=2)
- **E2.3**: Qwen-32B (4 GPUs, TP=4)

### E3: Expert Parallelism (MoE)
- **E3.1**: DeepSeek-V2-Lite (1 GPU, EP=1)
- **E3.2**: DeepSeek-V2-Lite (2 GPUs, EP=2)
- **E3.3**: DeepSeek-V2-Lite (4 GPUs, EP=4)
- **E3.4**: DeepSeek-V2-Lite (4 GPUs, TP=2, EP=2)

## Quick Start

### 1. Generate All Experiment Configurations

```bash
python experiments/generate_all_configs.py
```

This creates YAML configuration files in `experiments/configs/` for all experiments.

### 2. Run an Experiment on Perlmutter

```bash
# Run baseline experiment
sbatch experiments/slurm_scripts/run_E1.1.sh

# Run with Nsys profiling
sbatch --export=CONFIG=E1.3_llama-3.1-8b.yaml experiments/slurm_scripts/run_with_nsys.sh
```

### 3. Generate Workload Card

After an experiment completes, generate its workload card:

```bash
python experiments/generate_workload_card.py \
    --config experiments/configs/E1.1_llama-3.1-8b.yaml \
    --output-dir trace_collection/llama-3.1-8b-vllm-perlmutter-E1.1
```

### 4. Calculate Metrics

```bash
# Throughput
./scripts/get_throughput_tokens_sec.sh llama-3.1-8b-vllm-perlmutter-E1.1

# Iteration time
./scripts/get_iteration_wall_clock.sh llama-3.1-8b-vllm-perlmutter-E1.1

# Communication-computation overlap
./scripts/get_comm_comp_overlap.sh llama-3.1-8b-vllm-perlmutter-E1.1

# Number of NCCL communication calls
./scripts/get_coll_cal_num.sh llama-3.1-8b-vllm-perlmutter-E1.1
```

### 5. Analyze Results

After running multiple experiments, analyze and visualize:

```bash
python experiments/analyze_results.py
```

This generates:
- `experiments/results_summary.csv` - Summary table
- `experiments/scaling_analysis.png` - Scaling plots

## Modifying Experiments

### Add a New Experiment

1. Edit `experiments/generate_all_configs.py` and add your experiment to `EXPERIMENTS` dict
2. Run `python experiments/generate_all_configs.py`
3. Create corresponding SLURM script in `experiments/slurm_scripts/`

### Customize Profiling Parameters

Edit the experiment config YAML:

```yaml
warmup_iterations: 2      # Warmup before profiling
profile_iterations: 5     # Number of profiled iterations
data:
  batch_size: 4
  seq_len: 8192
  max_tokens: 128
```

## Trace Files

Each experiment generates:
- `torch_et_<rank>.json` - PyTorch execution trace
- `kineto_trace_<rank>.json` - Kineto GPU trace
- `nsys_<rank>.nsys-rep` - Nsys profiling report (if enabled)
- `timing_stats_<rank>.json` - Iteration timing statistics
- `workload_card.yaml` - Experiment metadata

## Metrics

Available metrics (see `tools/README.md` for full list):

1. **throughput_tokens_sec** - Tokens per second
2. **iteration_wall_clock** - Average iteration time
3. **comm_comp_overlap** - Communication-computation overlap %
4. **coll_call_num** - Number of NCCL calls

## Troubleshooting

### Model Download Issues

If models fail to download on Perlmutter:

```bash
# Pre-download models
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B
huggingface-cli download Qwen/Qwen-32B
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite
```

### Memory Issues

Reduce batch size or sequence length in config YAML:

```yaml
data:
  batch_size: 2    # Reduce from 4
  seq_len: 4096    # Reduce from 8192
```

### vLLM Not Found

Ensure vLLM is installed in your environment:

```bash
conda activate vllm-profiling
pip install vllm
```

## Next Steps

1. **Scale to 16 nodes** - Modify SLURM scripts to use multiple nodes
2. **Test NVSHMEM** - Compare with NCCL backend
3. **Add more metrics** - MFU, SM utilization, straggler detection
4. **Longer sequences** - Test with 16K, 32K sequence lengths

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Perlmutter User Guide](https://docs.nersc.gov/systems/perlmutter/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
