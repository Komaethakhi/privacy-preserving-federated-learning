import pytest
import pandas as pd
from dashboard.app import CREDENTIALS, load_results, load_model_comparison_csv, load_hospital_distribution_csv

import hashlib

def test_credentials_structure():
    assert "hospital_a" in CREDENTIALS
    assert "hospital_b" in CREDENTIALS
    assert "hospital_c" in CREDENTIALS
    assert "hospital_d" in CREDENTIALS
    assert "hospital_e" in CREDENTIALS
    assert "admin" in CREDENTIALS
    
    assert CREDENTIALS["hospital_a"]["password_hash"] == hashlib.sha256("hospital123".encode()).hexdigest()
    assert CREDENTIALS["admin"]["password_hash"] == hashlib.sha256("admin123".encode()).hexdigest()
    assert CREDENTIALS["admin"]["role"] == "System Administrator"


def test_results_data_integrity():
    comp_df = load_model_comparison_csv()
    hosp_df = load_hospital_distribution_csv()
    
    assert comp_df is not None
    assert len(comp_df) == 4
    assert set(comp_df["Model Paradigm"]).issubset({
        "Local-Only", "Centralized Baseline", "Federated (FedAvg)", "Federated + Privacy"
    })
    
    assert hosp_df is not None
    assert len(hosp_df) == 5
    assert set(hosp_df["hospital_name"]) == {
        "Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"
    }
