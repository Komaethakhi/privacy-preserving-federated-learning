import os
from typing import List, Dict, Any, Tuple
import time
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = hasattr(torch, "Tensor") and hasattr(torch, "randn_like")
except Exception:
    TORCH_AVAILABLE = False

from models.disease_model import DiseasePredictionMLP, NumPyDiseaseMLP, save_model_pt
from client.preprocessing import HospitalDataPreprocessor
from evaluation.metrics import calculate_classification_metrics

def run_centralized_experiment(
    hospital_data_dict: Dict[str, pd.DataFrame],
    in_features: int = 13,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    random_seed: int = 42,
    save_path: str = "models/centralized_model.pt"
) -> Tuple[Dict[str, Any], Any]:
    print(f"\n[EXPERIMENT: CENTRALIZED BASELINE] Training centralized model on pooled patient data...")
    start_time = time.time()

    df_pooled = pd.concat(list(hospital_data_dict.values()), ignore_index=True)
    
    preprocessor = HospitalDataPreprocessor()
    train_loader, test_loader, actual_in_features, stats = preprocessor.prepare_hospital_dataloaders(
        df_pooled, batch_size=batch_size, random_seed=random_seed
    )

    model = DiseasePredictionMLP(in_features=actual_in_features)
    used_torch = False

    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        try:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            model.train()
            for epoch in range(epochs):
                for X_b, y_b in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    logits = model(X_b)
                    all_logits.append(logits)
                    all_labels.append(y_b)

            cat_lg = torch.cat(all_logits, dim=0)
            cat_yb = torch.cat(all_labels, dim=0)
            probs = torch.sigmoid(cat_lg).cpu().numpy().flatten()
            y_true = cat_yb.cpu().numpy().flatten().astype(int)
            used_torch = True
        except Exception:
            used_torch = False

    if not used_torch:
        model = NumPyDiseaseMLP(in_features=actual_in_features)
        X_tr, y_tr = train_loader if isinstance(train_loader, tuple) else _extract_np(train_loader)
        for epoch in range(epochs):
            model.train_step(X_tr, y_tr, lr=learning_rate)

        X_te, y_te = test_loader if isinstance(test_loader, tuple) else _extract_np(test_loader)
        probs = model.predict_proba(X_te).flatten()
        y_true = y_te.flatten().astype(int)

    duration = time.time() - start_time
    preds = (probs >= 0.5).astype(int)

    metrics = calculate_classification_metrics(y_true, preds, probs)
    metrics["training_time_sec"] = duration
    metrics["communication_rounds"] = 0
    metrics["model_update_size_bytes"] = 0
    metrics["total_pooled_samples"] = len(df_pooled)

    # Save trained PyTorch model artifact to models/centralized_model.pt
    save_model_pt(model, save_path)
    print(f"  • Saved centralized model to '{save_path}'.")

    print(f"  Centralized Overall -> Acc: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} (Time: {duration:.2f}s)")
    return metrics, model

def _extract_np(loader):
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb.cpu().numpy() if (TORCH_AVAILABLE and isinstance(xb, torch.Tensor)) else np.array(xb))
        y_list.append(yb.cpu().numpy() if (TORCH_AVAILABLE and isinstance(yb, torch.Tensor)) else np.array(yb))
    return np.vstack(X_list), np.vstack(y_list)
