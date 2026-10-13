import os
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = hasattr(torch, "Tensor") and hasattr(torch, "randn_like")
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

if TORCH_AVAILABLE:
    class PyTorchDiseaseMLP(nn.Module):
        def __init__(self, in_features: int = 13, hidden_dim1: int = 32, hidden_dim2: int = 16, dropout_rate: float = 0.1):
            super(PyTorchDiseaseMLP, self).__init__()
            self.in_features = in_features
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, 1)
            )

        def forward(self, x: Any) -> torch.Tensor:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            return self.net(x)

        def predict_proba(self, x: Any) -> torch.Tensor:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.sigmoid(logits)
            if was_training:
                self.train()
            return probs

        def predict(self, x: Any, threshold: float = 0.5) -> torch.Tensor:
            probs = self.predict_proba(x)
            return (probs >= threshold).long()

class NumPyDiseaseMLP:
    def __init__(self, in_features: int = 13, hidden_dim1: int = 32, hidden_dim2: int = 16, random_seed: int = 42):
        np.random.seed(random_seed)
        self.in_features = in_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        
        self.W1 = np.random.randn(in_features, hidden_dim1) * np.sqrt(2.0 / in_features)
        self.b1 = np.zeros(hidden_dim1)
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1)
        self.b2 = np.zeros(hidden_dim2)
        self.W3 = np.random.randn(hidden_dim2, 1) * np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros(1)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "net.0.weight": self.W1.T.copy(),
            "net.0.bias": self.b1.copy(),
            "net.3.weight": self.W2.T.copy(),
            "net.3.bias": self.b2.copy(),
            "net.5.weight": self.W3.T.copy(),
            "net.5.bias": self.b3.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        def to_np(v):
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                return v.cpu().numpy()
            return np.array(v)

        self.W1 = to_np(state_dict["net.0.weight"]).T
        self.b1 = to_np(state_dict["net.0.bias"])
        self.W2 = to_np(state_dict["net.3.weight"]).T
        self.b2 = to_np(state_dict["net.3.bias"])
        self.W3 = to_np(state_dict["net.5.weight"]).T
        self.b3 = to_np(state_dict["net.5.bias"])

    def forward(self, X: np.ndarray) -> np.ndarray:
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.001):
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if TORCH_AVAILABLE and isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        logits = self.forward(X)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        
        dz3 = (probs - y) / m
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        eps = 1e-7
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        return loss


def DiseasePredictionMLP(in_features: int = 13, hidden_dim1: int = 32, hidden_dim2: int = 16, dropout_rate: float = 0.1):
    if TORCH_AVAILABLE:
        try:
            return PyTorchDiseaseMLP(in_features=in_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate)
        except Exception:
            return NumPyDiseaseMLP(in_features=in_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
    else:
        return NumPyDiseaseMLP(in_features=in_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)


def get_model_parameters(model: Any) -> Dict[str, Any]:
    return copy.deepcopy(model.state_dict())


def set_model_parameters(model: Any, state_dict: Dict[str, Any]) -> None:
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        converted_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                converted_dict[k] = torch.tensor(v, dtype=torch.float32)
            else:
                converted_dict[k] = v
        model.load_state_dict(converted_dict)
    else:
        model.load_state_dict(copy.deepcopy(state_dict))


def save_model_pt(model: Any, filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = get_model_parameters(model)
    if TORCH_AVAILABLE:
        torch.save(state, filepath)
    else:
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    return filepath


def load_model_pt(filepath: str, in_features: int = 13) -> Any:
    model = DiseasePredictionMLP(in_features=in_features)
    if TORCH_AVAILABLE:
        state = torch.load(filepath, map_location="cpu")
    else:
        import pickle
        with open(filepath, "rb") as f:
            state = pickle.load(f)
    set_model_parameters(model, state)
    return model


def compute_model_update(initial_state: Dict[str, Any], updated_state: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for key in initial_state.keys():
        v1 = updated_state[key]
        v2 = initial_state[key]
        if TORCH_AVAILABLE and isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            delta[key] = v1 - v2
        else:
            arr1 = v1.cpu().numpy() if (TORCH_AVAILABLE and isinstance(v1, torch.Tensor)) else np.array(v1)
            arr2 = v2.cpu().numpy() if (TORCH_AVAILABLE and isinstance(v2, torch.Tensor)) else np.array(v2)
            delta[key] = arr1 - arr2
    return delta


def apply_model_update(initial_state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    new_state = {}
    for key in initial_state.keys():
        v1 = initial_state[key]
        v2 = delta[key]
        if TORCH_AVAILABLE and isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            new_state[key] = v1 + v2
        else:
            arr1 = v1.cpu().numpy() if (TORCH_AVAILABLE and isinstance(v1, torch.Tensor)) else np.array(v1)
            arr2 = v2.cpu().numpy() if (TORCH_AVAILABLE and isinstance(v2, torch.Tensor)) else np.array(v2)
            new_state[key] = arr1 + arr2
    return new_state


def calculate_model_size_bytes(model: Any) -> int:
    state = get_model_parameters(model)
    total_bytes = 0
    for v in state.values():
        if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
            total_bytes += v.nelement() * v.element_size()
        else:
            arr = np.array(v)
            total_bytes += arr.nbytes
    return total_bytes
