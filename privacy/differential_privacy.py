import math
from typing import Dict, Tuple, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = hasattr(torch, "Tensor") and hasattr(torch, "randn_like")
except Exception:
    TORCH_AVAILABLE = False

try:
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    try:
        from opacus.accountants import RdpAccountant as RDPAccountant
        OPACUS_AVAILABLE = True
    except ImportError:
        OPACUS_AVAILABLE = False

class DifferentialPrivacyManager:
    """
    Differential Privacy-inspired L2 clipping and Gaussian noise mechanism
    with support for both formal Opacus RDP accounting and empirical baseline heuristics.
    """
    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 0.1, delta: float = 1e-5, enabled: bool = True):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.enabled = enabled
        self.mechanism_description = "DP noise + L2 clipping mechanism with Opacus RDP and heuristic accounting"

    def _tensor_sum_sq(self, p: Any) -> float:
        if TORCH_AVAILABLE and isinstance(p, torch.Tensor):
            return torch.sum(p ** 2).item()
        else:
            return float(np.sum(np.array(p) ** 2))

    def clip_model_update(self, update: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        if not self.enabled:
            total_norm_sq = sum(self._tensor_sum_sq(p) for p in update.values())
            return update, math.sqrt(total_norm_sq)

        total_norm_sq = sum(self._tensor_sum_sq(p) for p in update.values())
        l2_norm = math.sqrt(total_norm_sq)

        clip_factor = min(1.0, self.clip_norm / max(l2_norm, 1e-8))
        
        clipped_update = {}
        for k, p in update.items():
            clipped_update[k] = p * clip_factor

        return clipped_update, l2_norm

    def add_gaussian_noise(self, update: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.noise_multiplier <= 0.0:
            return update

        noisy_update = {}
        std_dev = self.noise_multiplier * self.clip_norm
        
        for k, p in update.items():
            if TORCH_AVAILABLE and isinstance(p, torch.Tensor):
                noise = torch.randn_like(p, dtype=p.dtype) * std_dev
                noisy_update[k] = p + noise
            else:
                arr = np.array(p)
                noise = np.random.normal(0, std_dev, size=arr.shape)
                noisy_update[k] = arr + noise

        return noisy_update

    def apply_privacy_protection(self, update: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        clipped_update, orig_norm = self.clip_model_update(update)
        protected_update = self.add_gaussian_noise(clipped_update)
        
        protected_norm_sq = sum(self._tensor_sum_sq(p) for p in protected_update.values())
        protected_norm = math.sqrt(protected_norm_sq)
        
        meta = {
            "original_l2_norm": orig_norm,
            "clipped_l2_norm": min(orig_norm, self.clip_norm) if self.enabled else orig_norm,
            "protected_l2_norm": protected_norm,
            "clip_norm_threshold": self.clip_norm,
            "noise_multiplier": self.noise_multiplier if self.enabled else 0.0,
            "dp_enabled": float(self.enabled)
        }
        return protected_update, meta

    def estimate_epsilon_heuristic(self, num_rounds: int, sample_ratio: float = 1.0) -> float:
        """Calculates an empirical baseline heuristic bound."""
        if not self.enabled or self.noise_multiplier <= 0.0:
            return float('inf')
        
        sigma = self.noise_multiplier
        T = num_rounds
        q = sample_ratio
        
        epsilon = (q * math.sqrt(T * 2 * math.log(1.25 / self.delta))) / sigma
        return round(epsilon, 3)

    def estimate_epsilon_dp_budget(self, num_rounds: int, sample_ratio: float = 1.0) -> float:
        """Alias for estimate_epsilon_heuristic for backward compatibility."""
        return self.estimate_epsilon_heuristic(num_rounds, sample_ratio)

    def estimate_epsilon_rdp(self, num_rounds: int, sample_ratio: float = 1.0) -> float:
        """Calculates formal Rényi Differential Privacy (RDP) epsilon bound using Opacus RDPAccountant."""
        if not self.enabled or self.noise_multiplier <= 0.0:
            return float('inf')

        if not OPACUS_AVAILABLE:
            return self.estimate_epsilon_heuristic(num_rounds, sample_ratio)

        accountant = RDPAccountant()
        for _ in range(num_rounds):
            accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=sample_ratio)

        eps = accountant.get_epsilon(delta=self.delta)
        return round(float(eps), 3)

