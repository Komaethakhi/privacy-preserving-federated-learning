from typing import List, Dict, Tuple, Any, Optional
import copy
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.disease_model import DiseasePredictionMLP, set_model_parameters, get_model_parameters
from server.aggregation import fedavg_aggregate_state_dicts
from client.hospital_client import HospitalClient
from privacy.differential_privacy import DifferentialPrivacyManager
from privacy.secure_aggregation import SecureAggregationSimulator
from evaluation.metrics import calculate_classification_metrics

class FederatedServer:
    def __init__(
        self,
        global_model: Any,
        hospital_clients: List[HospitalClient],
        privacy_manager: Optional[DifferentialPrivacyManager] = None,
        secure_aggregator: Optional[SecureAggregationSimulator] = None
    ):
        self.global_model = global_model
        self.clients = hospital_clients
        self.privacy_manager = privacy_manager if privacy_manager is not None else DifferentialPrivacyManager(enabled=False)
        self.secure_aggregator = secure_aggregator if secure_aggregator is not None else SecureAggregationSimulator(enabled=False)
        self.round_history: List[Dict[str, Any]] = []

    def evaluate_global_model(self) -> Dict[str, Any]:
        if TORCH_AVAILABLE and isinstance(self.global_model, torch.nn.Module):
            self.global_model.eval()
            all_logits = []
            all_labels = []
            criterion = nn.BCEWithLogitsLoss()
            total_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for client in self.clients:
                    for X_batch, y_batch in client.test_loader:
                        logits = self.global_model(X_batch)
                        loss = criterion(logits, y_batch)
                        total_loss += loss.item() * len(y_batch)
                        total_samples += len(y_batch)
                        
                        all_logits.append(logits)
                        all_labels.append(y_batch)

            if total_samples == 0:
                return {}

            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            probs = torch.sigmoid(cat_logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            y_true = cat_labels.cpu().numpy().flatten().astype(int)

            metrics = calculate_classification_metrics(y_true, preds, probs)
            metrics["loss"] = float(total_loss / total_samples)
            return metrics
        else:
            all_probs = []
            all_labels = []
            total_samples = 0
            total_loss = 0.0

            for client in self.clients:
                X_test, y_test = client.test_loader
                probs = self.global_model.predict_proba(X_test).flatten()
                y_true = y_test.flatten().astype(int)
                
                eps = 1e-7
                l = -np.mean(y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps))
                total_loss += l * len(y_true)
                total_samples += len(y_true)

                all_probs.append(probs)
                all_labels.append(y_true)

            cat_probs = np.concatenate(all_probs)
            cat_y = np.concatenate(all_labels)
            preds = (cat_probs >= 0.5).astype(int)

            metrics = calculate_classification_metrics(cat_y, preds, cat_probs)
            metrics["loss"] = float(total_loss / max(1, total_samples))
            return metrics

    def run_federated_round(self, round_idx: int, local_epochs: int = 3, learning_rate: float = 0.001) -> Dict[str, Any]:
        start_time = time.time()
        client_state_dicts = []
        client_sample_counts = []
        client_meta_list = []

        for client in self.clients:
            _, protected_state_dict, sample_count, round_meta = client.train_local_round(
                global_model=self.global_model,
                epochs=local_epochs,
                learning_rate=learning_rate,
                privacy_manager=self.privacy_manager
            )
            client_state_dicts.append(protected_state_dict)
            client_sample_counts.append(sample_count)
            client_meta_list.append(round_meta)

        aggregated_state_dict = fedavg_aggregate_state_dicts(
            client_state_dicts=client_state_dicts,
            client_sample_counts=client_sample_counts
        )

        set_model_parameters(self.global_model, aggregated_state_dict)

        eval_metrics = self.evaluate_global_model()
        round_duration = time.time() - start_time

        round_record = {
            "round": round_idx,
            "loss": eval_metrics.get("loss", 0.0),
            "accuracy": eval_metrics.get("accuracy", 0.0),
            "precision": eval_metrics.get("precision", 0.0),
            "recall": eval_metrics.get("recall", 0.0),
            "f1": eval_metrics.get("f1", 0.0),
            "roc_auc": eval_metrics.get("roc_auc", 0.0),
            "confusion_matrix": eval_metrics.get("confusion_matrix", []),
            "roc_curve": eval_metrics.get("roc_curve", {}),
            "round_duration_sec": round_duration,
            "participating_clients": len(self.clients),
            "client_metadata": client_meta_list
        }

        self.round_history.append(round_record)
        return round_record

    def run_full_training(self, num_rounds: int = 10, local_epochs: int = 3, learning_rate: float = 0.001) -> List[Dict[str, Any]]:
        print(f"\n[FEDERATED SERVER] Starting {num_rounds} FedAvg Communication Rounds with Privacy={self.privacy_manager.enabled}...")
        for r in range(1, num_rounds + 1):
            res = self.run_federated_round(round_idx=r, local_epochs=local_epochs, learning_rate=learning_rate)
            print(f"  Round {r:02d}/{num_rounds:02d} | Loss: {res['loss']:.4f} | Accuracy: {res['accuracy']*100:.2f}% | F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.4f} ({res['round_duration_sec']:.2f}s)")
        return self.round_history
