from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.disease_model import DiseasePredictionMLP, set_model_parameters, apply_model_update
from client.preprocessing import HospitalDataPreprocessor
from client.local_training import train_local_model
from privacy.differential_privacy import DifferentialPrivacyManager

class HospitalClient:
    """
    Simulated Hospital Client in the Federated Environment.
    """
    def __init__(self, hospital_name: str, df_local: pd.DataFrame, preprocessor: Optional[HospitalDataPreprocessor] = None, batch_size: int = 32):
        self.hospital_name = hospital_name
        self.raw_sample_count = len(df_local)
        
        self.preprocessor = preprocessor if preprocessor is not None else HospitalDataPreprocessor()
        self.train_loader, self.test_loader, self.in_features, self.stats = self.preprocessor.prepare_hospital_dataloaders(
            df_local, batch_size=batch_size
        )

    def train_local_round(
        self,
        global_model: Any,
        epochs: int = 3,
        learning_rate: float = 0.001,
        privacy_manager: Optional[DifferentialPrivacyManager] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]:
        updated_state_dict, raw_update, sample_count, local_loss = train_local_model(
            model=global_model,
            train_loader=self.train_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        initial_state = global_model.state_dict() if hasattr(global_model, "state_dict") else global_model.state_dict()

        if privacy_manager is not None:
            protected_update, dp_meta = privacy_manager.apply_privacy_protection(raw_update)
        else:
            protected_update = raw_update
            dp_meta = {"dp_enabled": 0.0, "original_l2_norm": 0.0}

        protected_state_dict = apply_model_update(initial_state, protected_update)

        round_meta = {
            "hospital_name": self.hospital_name,
            "sample_count": sample_count,
            "local_loss": local_loss,
            **dp_meta
        }

        return protected_update, protected_state_dict, sample_count, round_meta

    def evaluate_local(self, model: Any) -> Tuple[float, float]:
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            model.eval()
            criterion = nn.BCEWithLogitsLoss()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in self.test_loader:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    total_loss += loss.item() * len(y_batch)
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long()
                    correct += (preds == y_batch.long()).sum().item()
                    total += len(y_batch)

            avg_loss = total_loss / max(1, total)
            accuracy = correct / max(1, total)
            return float(avg_loss), float(accuracy)
        else:
            X_test, y_test = self.test_loader
            probs = model.predict_proba(X_test).flatten()
            preds = (probs >= 0.5).astype(int)
            y_true = y_test.flatten().astype(int)

            eps = 1e-7
            loss = -np.mean(y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps))
            accuracy = np.mean(preds == y_true)
            return float(loss), float(accuracy)
