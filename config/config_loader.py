import os
import json
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration with yaml or json/fallback parser."""
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
    except Exception:
        pass

    # Fallback default configuration dictionary if yaml module is missing or file unreadable
    return {
        "dataset": {
            "num_samples": 5000,
            "num_hospitals": 5,
            "hospital_names": ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"],
            "partition_type": "non_iid",
            "dirichlet_alpha": 0.5,
            "random_seed": 42
        },
        "model": {
            "model_type": "mlp",
            "hidden_dim1": 32,
            "hidden_dim2": 16,
            "dropout_rate": 0.1
        },
        "federated": {
            "num_rounds": 10,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "privacy": {
            "enabled": True,
            "clip_norm": 1.0,
            "noise_multiplier": 0.1,
            "delta": 0.00001
        },
        "paths": {
            "raw_data_path": "data/raw/synthetic_patients.csv",
            "hospital_data_dir": "data/hospital_data",
            "results_dir": "results",
            "plots_dir": "figures",
            "models_dir": "models",
            "model_save_path": "results/global_disease_model.pkl"
        }
    }
