import json
import os
from typing import Tuple, Optional

import yaml


PEAK_FLOPS_PER_GPU = 312e12  # A100 BF16 TFLOPS (tensor cores)
DEFAULT_PARAMS = 8e9  # Llama-8B approx params


def _load_batch_and_seq(config: dict) -> Tuple[int, int]:
    if "data" in config:
        return config["data"]["batch_size"], config["data"]["seq_len"]
    if "workload" in config and "data" in config["workload"]:
        data = config["workload"]["data"]
        return data["batch_size"], data["seq_len"]
    raise KeyError("Cannot find batch_size/seq_len in config")


def _load_gpu_count(config: dict) -> int:
    # New configs: parallelism block
    if "parallelism" in config:
        p = config["parallelism"]
        return p.get("tp", 1) * p.get("pp", 1) * p.get("dp_shard", 1) * p.get("dp_replicate", 1) * p.get("ep", 1)
    # Workload card fallback
    try:
        return int(config["workload"]["hardware"]["xpu_spec"]["total_count"])
    except Exception:
        return 1


def metric_cal(directory: str) -> float:
    """
    Estimate Model FLOPs Utilization (%) using throughput and model size.
    """
    timing_path = os.path.join(directory, "timing_stats_0.json")
    if not os.path.exists(timing_path):
        print(f"timing_stats_0.json not found in {directory}")
        return 0.0

    # Load timing stats
    with open(timing_path, "r") as fh:
        timing = json.load(fh)

    avg_iter_time = timing.get("avg_iteration_time", 0)
    if avg_iter_time <= 0:
        return 0.0

    # Load config (prefer config.yaml, fallback workload_card.yaml)
    cfg: Optional[dict] = None
    for name in ("config.yaml", "workload_card.yaml"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            with open(candidate, "r") as fh:
                cfg = yaml.safe_load(fh)
            break

    if cfg is None:
        print(f"No config or workload_card found in {directory}")
        return 0.0

    try:
        batch_size, seq_len = _load_batch_and_seq(cfg)
        gpus = max(1, _load_gpu_count(cfg))
    except Exception as exc:
        print(f"Error reading config for MFU: {exc}")
        return 0.0

    tokens_per_iter = batch_size * seq_len
    throughput = tokens_per_iter / avg_iter_time  # tokens/sec

    flops_per_token = 2 * DEFAULT_PARAMS
    total_flops = throughput * flops_per_token
    total_peak = gpus * PEAK_FLOPS_PER_GPU

    if total_peak <= 0:
        return 0.0

    return (total_flops / total_peak) * 100
