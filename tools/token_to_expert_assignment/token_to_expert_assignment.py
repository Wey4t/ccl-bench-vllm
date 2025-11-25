import json
import os
from typing import Dict, List
from collections import defaultdict


def metric_cal(directory: str) -> Dict[str, any]:
    """
    Calculate per-device token-to-expert assignment for Expert Parallelism (EP).

    This metric tracks how tokens are distributed across experts on each device
    in MoE (Mixture of Experts) models, useful for load balancing analysis.

    Args:
        directory (str): Path to the directory containing trace files.

    Returns:
        dict: Per-device token assignment statistics including:
            - 'device_assignments': dict mapping device_id to expert assignment counts
            - 'load_balance_std': standard deviation of token distribution
            - 'total_tokens_processed': total number of tokens
    """

    # Try to load workload card first to check if this is an MoE model
    workload_card_path = os.path.join(directory, "workload_card.yaml")

    try:
        import yaml
        if os.path.exists(workload_card_path):
            with open(workload_card_path, 'r') as f:
                workload_card = yaml.safe_load(f)
                is_moe = workload_card.get('workload', {}).get('model', {}).get('moe', False)
                if not is_moe:
                    print("Warning: This workload is not an MoE model")
                    return {
                        "device_assignments": {},
                        "load_balance_std": 0.0,
                        "total_tokens_processed": 0
                    }
    except Exception as e:
        print(f"Could not verify MoE model from workload card: {e}")

    # Look for PyTorch ET trace which may have expert routing information
    trace_file = os.path.join(directory, "torch_et_0.json")

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        device_expert_assignments = defaultdict(lambda: defaultdict(int))
        total_tokens = 0

        # Search for expert routing operations
        for node in trace_data.get("nodes", []):
            name = node.get("name", "").lower()

            # Look for MoE routing patterns
            if "expert" in name or "router" in name or "moe" in name:
                attrs = node.get("attrs", [])

                # Extract device ID
                device_id = 0
                for attr in attrs:
                    if attr.get("name") == "device":
                        device_id = attr.get("value", 0)
                        break

                # Extract expert ID and token count
                expert_id = None
                token_count = 1  # default to 1 token per operation

                for attr in attrs:
                    attr_name = attr.get("name", "").lower()
                    if "expert" in attr_name or "expert_id" in attr_name:
                        expert_id = attr.get("value")
                    elif "tokens" in attr_name or "batch" in attr_name:
                        token_count = attr.get("value", 1)

                if expert_id is not None:
                    device_expert_assignments[device_id][expert_id] += token_count
                    total_tokens += token_count

        if not device_expert_assignments:
            print("Warning: No expert assignment data found in trace")
            return {
                "device_assignments": {},
                "load_balance_std": 0.0,
                "total_tokens_processed": 0
            }

        # Calculate load balance statistics
        device_token_counts = []
        for device_id, expert_assignments in device_expert_assignments.items():
            device_total = sum(expert_assignments.values())
            device_token_counts.append(device_total)

        # Calculate standard deviation for load balance
        if len(device_token_counts) > 1:
            mean_tokens = sum(device_token_counts) / len(device_token_counts)
            variance = sum((x - mean_tokens) ** 2 for x in device_token_counts) / len(device_token_counts)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        # Convert defaultdict to regular dict for JSON serialization
        result = {
            "device_assignments": {
                str(device): dict(experts)
                for device, experts in device_expert_assignments.items()
            },
            "load_balance_std": std_dev,
            "total_tokens_processed": total_tokens
        }

        return result

    except FileNotFoundError:
        print(f"File not found: {trace_file}")
        return {
            "device_assignments": {},
            "load_balance_std": 0.0,
            "total_tokens_processed": 0
        }
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")
        return {
            "device_assignments": {},
            "load_balance_std": 0.0,
            "total_tokens_processed": 0
        }
    except Exception as e:
        print(f"Error calculating token-to-expert assignment: {e}")
        return {
            "device_assignments": {},
            "load_balance_std": 0.0,
            "total_tokens_processed": 0
        }
