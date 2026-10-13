import pytest
import torch
import numpy as np
from models.disease_model import DiseasePredictionMLP, get_model_parameters, set_model_parameters, save_model_pt, load_model_pt
from data.generate_data import generate_synthetic_healthcare_data
from client.preprocessing import HospitalDataPreprocessor
from client.local_training import train_local_model

def test_disease_mlp_forward():
    model = DiseasePredictionMLP(in_features=13)
    dummy_x = torch.randn(10, 13)
    logits = model(dummy_x)
    assert logits.shape == (10, 1)

def test_local_training():
    df = generate_synthetic_healthcare_data(num_samples=100, random_seed=42)
    preprocessor = HospitalDataPreprocessor()
    train_loader, _, in_feats, _ = preprocessor.prepare_hospital_dataloaders(df, batch_size=16)
    
    model = DiseasePredictionMLP(in_features=in_feats)
    updated_state, update_delta, sample_count, loss = train_local_model(
        model=model, train_loader=train_loader, epochs=1, learning_rate=0.001
    )
    
    assert sample_count > 0
    assert loss > 0.0
    assert "net.0.weight" in update_delta

def test_model_pt_save_and_load(tmp_path):
    model = DiseasePredictionMLP(in_features=13)
    save_path = str(tmp_path / "test_model.pt")
    save_model_pt(model, save_path)
    
    loaded_model = load_model_pt(save_path, in_features=13)
    dummy_x = torch.randn(5, 13)
    
    probs_orig = model.predict_proba(dummy_x)
    probs_loaded = loaded_model.predict_proba(dummy_x)
    
    assert torch.allclose(probs_orig, probs_loaded)
