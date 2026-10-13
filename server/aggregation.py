from typing import Dict, List, Any
import copy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def fedavg_aggregate_state_dicts(
    client_state_dicts: List[Dict[str, Any]],
    client_sample_counts: List[int]
) -> Dict[str, Any]:
    if len(client_state_dicts) == 0:
        raise ValueError("Cannot perform FedAvg on empty list of client state dicts.")
        
    n_total = sum(client_sample_counts)
    if n_total == 0:
        raise ValueError("Total client sample count is zero.")

    aggregated_state_dict = {}
    param_keys = client_state_dicts[0].keys()

    for key in param_keys:
        weighted_sum = None
        for i, state_dict in enumerate(client_state_dicts):
            weight = client_sample_counts[i] / float(n_total)
            p = state_dict[key]
            
            if TORCH_AVAILABLE and isinstance(p, torch.Tensor):
                param_tensor = p.float() * weight
            else:
                param_tensor = p * weight
            
            if weighted_sum is None:
                weighted_sum = param_tensor
            else:
                weighted_sum += param_tensor
                
        aggregated_state_dict[key] = weighted_sum

    return aggregated_state_dict


def fedavg_aggregate_updates(
    client_updates: List[Dict[str, Any]],
    client_sample_counts: List[int]
) -> Dict[str, Any]:
    return fedavg_aggregate_state_dicts(client_updates, client_sample_counts)
