import os
import pytest
import pandas as pd
import numpy as np
from data.generate_data import generate_synthetic_healthcare_data
from client.preprocessing import HospitalDataPreprocessor

def test_synthetic_data_generation():
    df = generate_synthetic_healthcare_data(num_samples=100, random_seed=42)
    assert len(df) == 100
    assert "disease_risk" in df.columns
    assert set(df["disease_risk"].unique()).issubset({0, 1})
    assert "patient_id" in df.columns

def test_preprocessor_pipeline():
    df = generate_synthetic_healthcare_data(num_samples=100, random_seed=42)
    preprocessor = HospitalDataPreprocessor()
    X, y = preprocessor.preprocess_df(df, is_train=True)
    
    assert X.shape[0] == 100
    assert X.shape[1] > 10
    assert len(y) == 100
    assert preprocessor.is_fitted
