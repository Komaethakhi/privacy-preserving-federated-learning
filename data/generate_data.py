import os
import numpy as np
import pandas as pd
from config.config_loader import load_config

def generate_synthetic_healthcare_data(num_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic tabular healthcare dataset for disease risk prediction.
    
    DISCLAIMER: This dataset is synthetically generated for academic research and 
    federated learning demonstration purposes only. It is not clinically validated.
    """
    np.random.seed(random_seed)
    
    age = np.random.randint(18, 85, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.48, 0.52])
    
    age_factor = (age - 18) / 67.0
    
    blood_pressure_sys = np.random.normal(110 + 25 * age_factor, 12, size=num_samples).clip(90, 190).astype(int)
    blood_pressure_dia = np.random.normal(70 + 15 * age_factor, 8, size=num_samples).clip(60, 115).astype(int)
    
    bmi = np.random.normal(25.5 + 4 * age_factor, 4.5, size=num_samples).clip(16.5, 45.0).round(1)
    glucose = np.random.normal(90 + 30 * age_factor + (bmi > 30) * 15, 25, size=num_samples).clip(70, 280).round(1)
    cholesterol = np.random.normal(180 + 35 * age_factor, 30, size=num_samples).clip(120, 350).round(1)
    heart_rate = np.random.normal(72, 10, size=num_samples).clip(50, 115).astype(int)
    
    smoking_status = np.random.choice(["Never", "Former", "Current"], size=num_samples, p=[0.55, 0.25, 0.20])
    family_history = np.random.choice([0, 1], size=num_samples, p=[0.65, 0.35])
    
    hypertension = ((blood_pressure_sys >= 130) | (blood_pressure_dia >= 80)).astype(int)
    diabetes = ((glucose >= 126) | (bmi >= 32)).astype(int)
    
    smoke_val = np.where(smoking_status == "Current", 1.2, np.where(smoking_status == "Former", 0.5, 0.0))
    gender_val = np.where(gender == "Male", 0.3, 0.0)
    
    logit = (
        -2.5
        + 0.04 * (age - 45)
        + 0.02 * (blood_pressure_sys - 120)
        + 0.015 * (glucose - 100)
        + 0.01 * (cholesterol - 200)
        + 0.05 * (bmi - 25)
        + 0.8 * smoke_val
        + 0.7 * family_history
        + 0.9 * hypertension
        + 1.1 * diabetes
        + gender_val
    )

    
    prob = 1.0 / (1.0 + np.exp(-logit))
    disease_risk = (np.random.uniform(0, 1, size=num_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "patient_id": [f"PAT_{i+1:05d}" for i in range(num_samples)],
        "age": age,
        "gender": gender,
        "blood_pressure_sys": blood_pressure_sys,
        "blood_pressure_dia": blood_pressure_dia,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "smoking_status": smoking_status,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "family_history": family_history,
        "disease_risk": disease_risk
    })
    
    return df

def save_raw_dataset(config_path: str = "config/config.yaml") -> str:
    config = load_config(config_path)
        
    num_samples = config["dataset"]["num_samples"]
    random_seed = config["dataset"]["random_seed"]
    output_path = config["paths"]["raw_data_path"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_healthcare_data(num_samples=num_samples, random_seed=random_seed)
    df.to_csv(output_path, index=False)
    print(f"[DATA GENERATOR] Successfully generated {num_samples} synthetic patient records at '{output_path}'.")
    return output_path

if __name__ == "__main__":
    save_raw_dataset()
