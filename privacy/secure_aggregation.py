import hashlib
from typing import Dict, List, Any, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SecureAggregationSimulator:
    def __init__(self, num_hospitals: int = 5, enabled: bool = False):
        self.num_hospitals = num_hospitals
        self.enabled = enabled
        self.status_message = "Conceptual/simulated secure aggregation module; production cryptographic implementation is future work."

    def generate_pairwise_masks(self, client_ids: List[str], sample_update: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not self.enabled:
            empty_masks = {}
            for cid in client_ids:
                empty_masks[cid] = {}
                for k, v in sample_update.items():
                    if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                        empty_masks[cid][k] = torch.zeros_like(v)
                    else:
                        empty_masks[cid][k] = np.zeros_like(v)
            return empty_masks

        client_masks = {}
        for cid in client_ids:
            client_masks[cid] = {}
            for k, v in sample_update.items():
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    client_masks[cid][k] = torch.zeros_like(v)
                else:
                    client_masks[cid][k] = np.zeros_like(v)

        n = len(client_ids)
        for i in range(n):
            for j in range(i + 1, n):
                cid_i = client_ids[i]
                cid_j = client_ids[j]
                
                for k, v in sample_update.items():
                    seed_str = f"{cid_i}_{cid_j}_{k}"
                    pairwise_seed = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
                    
                    if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                        gen = torch.Generator(device=v.device if hasattr(v, 'device') else 'cpu')
                        gen.manual_seed(pairwise_seed)
                        mask = torch.randn(v.shape, generator=gen, dtype=v.dtype, device=v.device) * 0.05
                    else:
                        rng = np.random.default_rng(pairwise_seed)
                        mask = rng.normal(0, 0.05, size=np.array(v).shape)

                    client_masks[cid_i][k] += mask
                    client_masks[cid_j][k] -= mask

        return client_masks

    def apply_secure_mask(self, client_id: str, update: Dict[str, Any], all_masks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not self.enabled or client_id not in all_masks:
            return update
        
        masked_update = {}
        client_mask = all_masks[client_id]
        for k, v in update.items():
            masked_update[k] = v + client_mask[k]
        return masked_update

    def verify_mask_cancellation(
        self,
        masked_updates: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        unmasked_updates: List[Dict[str, Any]] = None
    ) -> bool:
        """
        Verifies that pairwise masks cancel out during aggregation.
        - If unmasked_updates is provided: verifies sum(masked_updates) == sum(unmasked_updates).
        - If unmasked_updates is None: verifies sum(masks) == 0 (where masked_updates is either a dict of masks or a list of masks/updates).
        """
        if not self.enabled:
            return True
            
        if not masked_updates:
            return True

        if isinstance(masked_updates, dict):
            masks_list = list(masked_updates.values())
        else:
            masks_list = masked_updates

        if not masks_list:
            return True

        if unmasked_updates is not None:
            if len(masks_list) != len(unmasked_updates):
                return False
            for k in masks_list[0].keys():
                if TORCH_AVAILABLE and isinstance(masks_list[0][k], torch.Tensor):
                    sum_masked = torch.stack([m[k] for m in masks_list]).sum(dim=0)
                    sum_unmasked = torch.stack([u[k] for u in unmasked_updates]).sum(dim=0)
                    if not torch.allclose(sum_masked, sum_unmasked, atol=1e-4):
                        return False
                else:
                    sum_masked = np.sum([np.array(m[k]) for m in masks_list], axis=0)
                    sum_unmasked = np.sum([np.array(u[k]) for u in unmasked_updates], axis=0)
                    if not np.allclose(sum_masked, sum_unmasked, atol=1e-4):
                        return False
            return True

        # Check if masks sum to zero
        for k in masks_list[0].keys():
            if TORCH_AVAILABLE and isinstance(masks_list[0][k], torch.Tensor):
                sum_val = torch.stack([m[k] for m in masks_list]).sum(dim=0)
                if not torch.allclose(sum_val, torch.zeros_like(sum_val), atol=1e-4):
                    return False
            else:
                sum_val = np.sum([np.array(m[k]) for m in masks_list], axis=0)
                if not np.allclose(sum_val, np.zeros_like(sum_val), atol=1e-4):
                    return False

        return True

