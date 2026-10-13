from typing import List, Dict, Any, Optional, Tuple
import time
import copy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.disease_model import DiseasePredictionMLP, calculate_model_size_bytes, save_model_pt
from client.hospital_client import HospitalClient
from server.federated_server import FederatedServer
from privacy.differential_privacy import DifferentialPrivacyManager
from privacy.secure_aggregation import SecureAggregationSimulator

def run_federated_experiment(
    hospital_clients: List[HospitalClient],
    in_features: int = 13,
    num_rounds: int = 10,
    local_epochs: int = 3,
    learning_rate: float = 0.001,
    enable_dp: bool = False,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.1,
    delta: float = 1e-5,
    save_model_path: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    label = "FEDERATED + DP NOISE" if enable_dp else "FEDERATED (STANDARD FEDAVG)"
    print(f"\n[EXPERIMENT: {label}] Initializing FL Server across {len(hospital_clients)} hospitals...")
    
    start_time = time.time()
    global_model = DiseasePredictionMLP(in_features=in_features)
    
    privacy_manager = DifferentialPrivacyManager(
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        delta=delta,
        enabled=enable_dp
    )
    
    secure_aggregator = SecureAggregationSimulator(
        num_hospitals=len(hospital_clients),
        enabled=False
    )

    server = FederatedServer(
        global_model=global_model,
        hospital_clients=hospital_clients,
        privacy_manager=privacy_manager,
        secure_aggregator=secure_aggregator
    )

    history = server.run_full_training(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )

    duration = time.time() - start_time
    final_round_metrics = history[-1] if len(history) > 0 else {}
    
    update_size = calculate_model_size_bytes(global_model)
    estimated_eps = privacy_manager.estimate_epsilon_dp_budget(num_rounds=num_rounds) if enable_dp else float('inf')

    summary_metrics = {
        "accuracy": final_round_metrics.get("accuracy", 0.0),
        "precision": final_round_metrics.get("precision", 0.0),
        "recall": final_round_metrics.get("recall", 0.0),
        "f1": final_round_metrics.get("f1", 0.0),
        "roc_auc": final_round_metrics.get("roc_auc", 0.0),
        "loss": final_round_metrics.get("loss", 0.0),
        "training_time_sec": duration,
        "communication_rounds": num_rounds,
        "model_update_size_bytes": update_size,
        "dp_enabled": enable_dp,
        "clip_norm": clip_norm if enable_dp else None,
        "noise_multiplier": noise_multiplier if enable_dp else None,
        "estimated_epsilon": estimated_eps,
        "confusion_matrix": final_round_metrics.get("confusion_matrix", []),
        "roc_curve": final_round_metrics.get("roc_curve", {})
    }

    if save_model_path:
        save_model_pt(global_model, save_model_path)
        print(f"  • Saved model checkpoint to '{save_model_path}'.")

    print(f"  {label} Final -> Acc: {summary_metrics['accuracy']*100:.2f}% | F1: {summary_metrics['f1']:.4f} | ROC-AUC: {summary_metrics['roc_auc']:.4f} (Rounds: {num_rounds}, Time: {duration:.2f}s)")
    return summary_metrics, history, global_model
