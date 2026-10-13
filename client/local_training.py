from typing import Dict, Tuple, Any
import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = hasattr(torch, "Tensor") and hasattr(torch, "randn_like")
except Exception:
    TORCH_AVAILABLE = False

from models.disease_model import (
    DiseasePredictionMLP, get_model_parameters, set_model_parameters, compute_model_update, NumPyDiseaseMLP
)

def train_local_model(
    model: Any,
    train_loader: Any,
    epochs: int = 3,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> Tuple[Dict[str, Any], Dict[str, Any], int, float]:
    """
    Executes local model training on a single hospital's isolated dataset.
    Supports both PyTorch and NumPy backends with automatic error recovery.
    """
    model = copy.deepcopy(model)
    initial_state_dict = get_model_parameters(model)

    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        try:
            model.to(device)
            model.train()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            total_loss = 0.0
            total_batches = 0
            sample_count = len(train_loader.dataset)

            for epoch in range(epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    total_batches += 1

                total_loss += epoch_loss

            avg_loss = total_loss / max(1, total_batches)
            updated_state_dict = get_model_parameters(model)
            weight_update = compute_model_update(initial_state_dict, updated_state_dict)
            return updated_state_dict, weight_update, sample_count, float(avg_loss)
        except Exception:
            numpy_model = NumPyDiseaseMLP(in_features=getattr(model, "in_features", 13))
            numpy_model.load_state_dict(initial_state_dict)
            return _train_numpy(numpy_model, train_loader, epochs, learning_rate, initial_state_dict)
    else:
        return _train_numpy(model, train_loader, epochs, learning_rate, initial_state_dict)

def _train_numpy(model: Any, train_loader: Any, epochs: int, learning_rate: float, initial_state_dict: Dict[str, Any]):
    if isinstance(train_loader, tuple):
        X_train, y_train = train_loader
    else:
        X_list, y_list = [], []
        for xb, yb in train_loader:
            X_list.append(xb.cpu().numpy() if TORCH_AVAILABLE and isinstance(xb, torch.Tensor) else np.array(xb))
            y_list.append(yb.cpu().numpy() if TORCH_AVAILABLE and isinstance(yb, torch.Tensor) else np.array(yb))
        X_train = np.vstack(X_list)
        y_train = np.vstack(y_list)

    sample_count = len(X_train)
    total_loss = 0.0
    batch_size = 32
    num_batches = int(np.ceil(sample_count / batch_size))

    for epoch in range(epochs):
        indices = np.arange(sample_count)
        np.random.shuffle(indices)
        epoch_loss = 0.0
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            X_b, y_b = X_train[batch_idx], y_train[batch_idx]
            l = model.train_step(X_b, y_b, lr=learning_rate)
            epoch_loss += l

        total_loss += (epoch_loss / max(1, num_batches))

    avg_loss = total_loss / max(1, epochs)
    updated_state_dict = get_model_parameters(model)
    weight_update = compute_model_update(initial_state_dict, updated_state_dict)
    return updated_state_dict, weight_update, sample_count, float(avg_loss)
