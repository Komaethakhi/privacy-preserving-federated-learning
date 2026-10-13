from typing import List, Dict, Any
import copy
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = hasattr(torch, "Tensor") and hasattr(torch, "randn_like")
except Exception:
    TORCH_AVAILABLE = False

from models.disease_model import DiseasePredictionMLP, NumPyDiseaseMLP
from client.hospital_client import HospitalClient
from evaluation.metrics import calculate_classification_metrics

def run_local_only_experiment(
    hospital_clients: List[HospitalClient],
    in_features: int = 13,
    epochs: int = 15,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    print(f"\n[EXPERIMENT: LOCAL-ONLY] Training {len(hospital_clients)} independent hospital models for {epochs} local epochs...")
    start_time = time.time()
    
    hospital_results = {}
    all_test_probs = []
    all_test_labels = []

    for client in hospital_clients:
        model = DiseasePredictionMLP(in_features=in_features)
        used_torch = False

        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            try:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.BCEWithLogitsLoss()

                model.train()
                for epoch in range(epochs):
                    for X_b, y_b in client.train_loader:
                        optimizer.zero_grad()
                        logits = model(X_b)
                        loss = criterion(logits, y_b)
                        loss.backward()
                        optimizer.step()

                model.eval()
                client_logits = []
                client_labels = []
                with torch.no_grad():
                    for X_b, y_b in client.test_loader:
                        lg = model(X_b)
                        client_logits.append(lg)
                        client_labels.append(y_b)

                cat_lg = torch.cat(client_logits, dim=0)
                cat_yb = torch.cat(client_labels, dim=0)
                probs = torch.sigmoid(cat_lg).cpu().numpy().flatten()
                y_true = cat_yb.cpu().numpy().flatten().astype(int)
                used_torch = True
            except Exception:
                used_torch = False

        if not used_torch:
            numpy_model = NumPyDiseaseMLP(in_features=in_features)
            X_tr, y_tr = client.train_loader if isinstance(client.train_loader, tuple) else _extract_np(client.train_loader)
            for epoch in range(epochs):
                numpy_model.train_step(X_tr, y_tr, lr=learning_rate)

            X_te, y_te = client.test_loader if isinstance(client.test_loader, tuple) else _extract_np(client.test_loader)
            probs = numpy_model.predict_proba(X_te).flatten()
            y_true = y_te.flatten().astype(int)

        preds = (probs >= 0.5).astype(int)
        local_metrics = calculate_classification_metrics(y_true, preds, probs)
        hospital_results[client.hospital_name] = local_metrics

        all_test_probs.append(probs)
        all_test_labels.append(y_true)

    duration = time.time() - start_time

    cat_all_probs = np.concatenate(all_test_probs)
    cat_all_yb = np.concatenate(all_test_labels)
    all_preds = (cat_all_probs >= 0.5).astype(int)

    overall_metrics = calculate_classification_metrics(cat_all_yb, all_preds, cat_all_probs)
    overall_metrics["training_time_sec"] = duration
    overall_metrics["hospital_details"] = hospital_results
    overall_metrics["communication_rounds"] = 0
    overall_metrics["model_update_size_bytes"] = 0

    print(f"  Local-Only Overall -> Acc: {overall_metrics['accuracy']*100:.2f}% | F1: {overall_metrics['f1']:.4f} | ROC-AUC: {overall_metrics['roc_auc']:.4f} (Time: {duration:.2f}s)")
    return overall_metrics

def _extract_np(loader):
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb.cpu().numpy() if (TORCH_AVAILABLE and isinstance(xb, torch.Tensor)) else np.array(xb))
        y_list.append(yb.cpu().numpy() if (TORCH_AVAILABLE and isinstance(yb, torch.Tensor)) else np.array(yb))
    return np.vstack(X_list), np.vstack(y_list)
