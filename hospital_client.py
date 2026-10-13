from typing import Dict, Tuple, Any, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.disease_model import DiseasePredictionMLP, set_model_parameters, apply_model_update
from client.preprocessing import HospitalDataPreprocessor
from client.local_training import train_local_model
from privacy.differential_privacy import DifferentialPrivacyManager

class HospitalClient:
    """
    Simulated Hospital Client in the Federated Environment.
    
    KEY PRIVACY GUARANTEE:
    Raw patient dataset (df_local) remains strictly encapsulated inside this object.
    Only model weights/updates or privacy-enhanced updates are transmitted to the server.
    """
    def __init__(self, hospital_name: str, df_local: pd.DataFrame, preprocessor: Optional[HospitalDataPreprocessor] = None, batch_size: int = 32):
        self.hospital_name = hospital_name
        self.raw_sample_count = len(df_local)
        
        # Preprocess and split local dataset
        self.preprocessor = preprocessor if preprocessor is not None else HospitalDataPreprocessor()
        self.train_loader, self.test_loader, self.in_features, self.stats = self.preprocessor.prepare_hospital_dataloaders(
            df_local, batch_size=batch_size
        )

    def train_local_round(
        self,
        global_model: nn.Module,
        epochs: int = 3,
        learning_rate: float = 0.005,
        privacy_manager: Optional[DifferentialPrivacyManager] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, Dict[str, Any]]:
        """
        Trains the local model copy on isolated hospital data and applies privacy protection.
        
        Returns:
        - protected_update: L2-clipped and Gaussian-noised weight update (Delta)
        - protected_state_dict: Global_w + protected_update
        - sample_count (n_k): Number of local training samples
        - round_meta: Local loss, DP norms, sample counts
        """
        # Train locally starting from global model parameters
        updated_state_dict, raw_update, sample_count, local_loss = train_local_model(
            model=global_model,
            train_loader=self.train_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        initial_state = global_model.state_dict()

        # Apply Differential Privacy Layer if provided
        if privacy_manager is not None:
            protected_update, dp_meta = privacy_manager.apply_privacy_protection(raw_update)
        else:
            protected_update = raw_update
            dp_meta = {"dp_enabled": 0.0, "original_l2_norm": 0.0}

        # Calculate new state dict with privacy-enhanced update
        protected_state_dict = apply_model_update(initial_state, protected_update)

        round_meta = {
            "hospital_name": self.hospital_name,
            "sample_count": sample_count,
            "local_loss": local_loss,
            **dp_meta
        }

        return protected_update, protected_state_dict, sample_count, round_meta

    def evaluate_local(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluates model performance on local test dataset."""
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
