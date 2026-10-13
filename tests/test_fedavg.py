import pytest
import torch
from server.aggregation import fedavg_aggregate_state_dicts

def test_fedavg_weighted_aggregation():
    # Two simple client state dicts
    dict1 = {"w": torch.tensor([2.0, 4.0])}
    dict2 = {"w": torch.tensor([6.0, 12.0])}
    
    # Client 1 has 100 samples, Client 2 has 300 samples (total 400)
    # Weights: 100/400 = 0.25, 300/400 = 0.75
    # Expected result: 0.25 * [2, 4] + 0.75 * [6, 12] = [0.5 + 4.5, 1.0 + 9.0] = [5.0, 10.0]
    agg = fedavg_aggregate_state_dicts([dict1, dict2], [100, 300])
    
    expected = torch.tensor([5.0, 10.0])
    assert torch.allclose(agg["w"], expected)
