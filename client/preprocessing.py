from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler as SklearnScaler
    from sklearn.model_selection import train_test_split as sklearn_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class HealthcareDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]

class PureNumPyScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

def numpy_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int = 42):
    np.random.seed(random_seed)
    n = len(X)
    test_len = int(n * test_size)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    test_idx = indices[:test_len]
    train_idx = indices[test_len:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class HospitalDataPreprocessor:
    def __init__(self, scaler: Optional[Any] = None):
        if scaler is not None:
            self.scaler = scaler
            self.is_fitted = True
        else:
            self.scaler = SklearnScaler() if SKLEARN_AVAILABLE else PureNumPyScaler()
            self.is_fitted = False
        self.feature_names: List[str] = []

    def preprocess_df(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        required_cols = [
            "age", "gender", "blood_pressure_sys", "blood_pressure_dia",
            "glucose", "cholesterol", "bmi", "heart_rate", "smoking_status",
            "hypertension", "diabetes", "family_history", "disease_risk"
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        df_clean = df.copy()
        
        num_cols = ["age", "blood_pressure_sys", "blood_pressure_dia", "glucose", "cholesterol", "bmi", "heart_rate"]
        for col in num_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        df_encoded = pd.get_dummies(df_clean, columns=["gender", "smoking_status"], drop_first=False)
        y = df_encoded["disease_risk"].values.astype(np.float32)
        
        drop_cols = ["patient_id", "disease_risk"]
        X_df = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])
        
        if self.is_fitted and len(self.feature_names) > 0:
            for col in self.feature_names:
                if col not in X_df.columns:
                    X_df[col] = 0.0
            X_df = X_df[self.feature_names]
        else:
            self.feature_names = list(X_df.columns)

        X_num = X_df[num_cols].values
        X_cat = X_df.drop(columns=num_cols).values

        if is_train and not self.is_fitted:
            X_num_scaled = self.scaler.fit_transform(X_num)
            self.is_fitted = True
        else:
            X_num_scaled = self.scaler.transform(X_num)

        X_processed = np.hstack([X_num_scaled, X_cat]).astype(np.float32)
        return X_processed, y

    def prepare_hospital_dataloaders(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        batch_size: int = 32, 
        random_seed: int = 42
    ) -> Tuple[Any, Any, int, Dict[str, float]]:
        X, y = self.preprocess_df(df, is_train=True)
        
        if SKLEARN_AVAILABLE:
            X_train, X_test, y_train, y_test = sklearn_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y if len(np.unique(y)) > 1 else None
            )
        else:
            X_train, X_test, y_train, y_test = numpy_train_test_split(
                X, y, test_size=test_size, random_seed=random_seed
            )
        
        stats = {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_cases": int(y.sum()),
            "prevalence_pct": round((y.sum() / len(y)) * 100.0, 2)
        }

        if TORCH_AVAILABLE:
            train_dataset = HealthcareDataset(X_train, y_train)
            test_dataset = HealthcareDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader, X.shape[1], stats
        else:
            train_tuple = (X_train, y_train.reshape(-1, 1))
            test_tuple = (X_test, y_test.reshape(-1, 1))
            return train_tuple, test_tuple, X.shape[1], stats
