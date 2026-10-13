import pytest
import math
import torch
from privacy.differential_privacy import DifferentialPrivacyManager

def test_l2_norm_clipping():
    dp = DifferentialPrivacyManager(clip_norm=1.0, noise_multiplier=0.0, enabled=True)
    large_update = {"w": torch.tensor([3.0, 4.0])}  # Norm is 5.0
    
    clipped, norm = dp.clip_model_update(large_update)
    clipped_norm = math.sqrt(torch.sum(clipped["w"] ** 2).item())
    
    assert norm == 5.0
    assert math.isclose(clipped_norm, 1.0, abs_tol=1e-5)

def test_gaussian_noise_addition():
    dp = DifferentialPrivacyManager(clip_norm=1.0, noise_multiplier=0.5, enabled=True)
    update = {"w": torch.zeros(100)}
    
    protected, meta = dp.apply_privacy_protection(update)
    assert meta["dp_enabled"] == 1.0
    assert torch.std(protected["w"]).item() > 0.1

def test_secure_aggregation_mask_cancellation():
    from privacy.secure_aggregation import SecureAggregationSimulator
    sec_agg = SecureAggregationSimulator(num_hospitals=3, enabled=True)
    client_ids = ["h1", "h2", "h3"]
    sample_update = {"w": torch.tensor([1.0, 2.0, 3.0])}

    masks = sec_agg.generate_pairwise_masks(client_ids, sample_update)
    assert sec_agg.verify_mask_cancellation(masks) is True

    # Test applying masks and verifying sum
    unmasked = [
        {"w": torch.tensor([1.0, 2.0, 3.0])},
        {"w": torch.tensor([4.0, 5.0, 6.0])},
        {"w": torch.tensor([7.0, 8.0, 9.0])},
    ]
    masked = [sec_agg.apply_secure_mask(cid, u, masks) for cid, u in zip(client_ids, unmasked)]
    assert sec_agg.verify_mask_cancellation(masked, unmasked) is True

    # Corrupt a mask and verify failure
    corrupted_masks = {cid: {k: v.clone() for k, v in m.items()} for cid, m in masks.items()}
    corrupted_masks["h1"]["w"] += 1.0
    assert sec_agg.verify_mask_cancellation(corrupted_masks) is False

def test_secure_aggregation_rng_isolation():
    import numpy as np
    from privacy.secure_aggregation import SecureAggregationSimulator
    sec_agg = SecureAggregationSimulator(num_hospitals=3, enabled=True)
    
    # Record torch and numpy RNG states before generate_pairwise_masks
    torch_state_before = torch.get_rng_state()
    np_state_before = np.random.get_state()

    client_ids = ["h1", "h2"]
    sample_update = {"w": torch.tensor([1.0, 2.0])}
    sec_agg.generate_pairwise_masks(client_ids, sample_update)

    torch_state_after = torch.get_rng_state()
    np_state_after = np.random.get_state()

    assert torch.equal(torch_state_before, torch_state_after)
    assert np.all(np_state_before[1] == np_state_after[1])

def test_epsilon_accounting():
    dp = DifferentialPrivacyManager(clip_norm=1.0, noise_multiplier=0.1, delta=1e-5, enabled=True)
    eps_heur = dp.estimate_epsilon_heuristic(num_rounds=10)
    eps_rdp = dp.estimate_epsilon_rdp(num_rounds=10)
    eps_compat = dp.estimate_epsilon_dp_budget(num_rounds=10)

    assert isinstance(eps_heur, float)
    assert isinstance(eps_rdp, float)
    assert eps_heur > 0.0
    assert eps_rdp > 0.0
    assert eps_heur == eps_compat

    # Disabled DP should return inf
    dp_disabled = DifferentialPrivacyManager(enabled=False)
    assert dp_disabled.estimate_epsilon_rdp(num_rounds=10) == float('inf')
    assert dp_disabled.estimate_epsilon_heuristic(num_rounds=10) == float('inf')


