import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from config.config_loader import load_config

def _dirichlet_partition(
    df: pd.DataFrame, 
    hospital_names: List[str], 
    dirichlet_alpha: float, 
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    np.random.seed(random_seed)
    num_hospitals = len(hospital_names)
    hospital_indices = {name: [] for name in hospital_names}
    
    classes = df["disease_risk"].unique()
    for c in sorted(classes):
        class_idx = df[df["disease_risk"] == c].index.values
        np.random.shuffle(class_idx)
        
        # Draw class proportions across hospitals from Dirichlet distribution
        proportions = np.random.dirichlet([dirichlet_alpha] * num_hospitals)
        min_p = 0.05
        proportions = (1.0 - min_p * num_hospitals) * proportions + min_p
        proportions /= proportions.sum()

        # Convert proportions to integer counts
        counts = (proportions * len(class_idx)).astype(int)
        diff = len(class_idx) - counts.sum()
        for i in range(diff):
            counts[i % num_hospitals] += 1
            
        start = 0
        for i, name in enumerate(hospital_names):
            end = start + counts[i]
            hospital_indices[name].extend(class_idx[start:end])
            start = end

            
    hospital_datasets = {}
    for name in hospital_names:
        h_idx = np.array(hospital_indices[name])
        np.random.shuffle(h_idx)
        hospital_datasets[name] = df.loc[h_idx].reset_index(drop=True)
        
    return hospital_datasets


def partition_dataset(
    df: pd.DataFrame, 
    hospital_names: List[str], 
    partition_type: str = "non_iid",
    dirichlet_alpha: float = 0.5,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    np.random.seed(random_seed)
    num_hospitals = len(hospital_names)
    hospital_datasets = {}
    
    if partition_type == "iid":
        shuffled_df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        splits = np.array_split(shuffled_df, num_hospitals)
        for idx, name in enumerate(hospital_names):
            hospital_datasets[name] = splits[idx].reset_index(drop=True)
            
    elif partition_type in ("non_iid", "dirichlet"):
        # Dirichlet-based non-IID label distribution (McMahan/Hsu et al.)
        hospital_datasets = _dirichlet_partition(
            df, hospital_names, dirichlet_alpha, random_seed
        )
        
    elif partition_type == "demographic":
        # Hand-crafted demographic cohort skew logic (age, glucose, BP cohorts)
        indices = np.arange(len(df))
        hospital_indices = {name: [] for name in hospital_names}
        df_sorted = df.copy()
        
        # Hospital A: Geriatric focus (higher age)
        prob_A = (df_sorted["age"] >= 50).astype(float) + 0.1
        prob_A /= prob_A.sum()
        idx_A = np.random.choice(indices, size=int(len(df) * 0.22), replace=False, p=prob_A)
        hospital_indices[hospital_names[0]] = idx_A
        remaining_indices = np.setdiff1d(indices, idx_A)
        
        # Hospital B: Metabolic focus (higher glucose & diabetes)
        df_rem = df_sorted.loc[remaining_indices]
        prob_B = (df_rem["glucose"] > 110).astype(float) + (df_rem["bmi"] > 28).astype(float) + 0.1
        prob_B /= prob_B.sum()
        idx_B = np.random.choice(remaining_indices, size=int(len(df) * 0.20), replace=False, p=prob_B)
        hospital_indices[hospital_names[1]] = idx_B
        remaining_indices = np.setdiff1d(remaining_indices, idx_B)
        
        # Hospital C: Younger demographic (lower age/BMI)
        df_rem = df_sorted.loc[remaining_indices]
        prob_C = (df_rem["age"] < 40).astype(float) + 0.1
        prob_C /= prob_C.sum()
        idx_C = np.random.choice(remaining_indices, size=int(len(df) * 0.18), replace=False, p=prob_C)
        hospital_indices[hospital_names[2]] = idx_C
        remaining_indices = np.setdiff1d(remaining_indices, idx_C)
        
        # Hospital D: Cardiovascular focus (higher blood pressure & cholesterol)
        df_rem = df_sorted.loc[remaining_indices]
        prob_D = (df_rem["blood_pressure_sys"] > 130).astype(float) + (df_rem["cholesterol"] > 220).astype(float) + 0.1
        prob_D /= prob_D.sum()
        idx_D = np.random.choice(remaining_indices, size=int(len(df) * 0.20), replace=False, p=prob_D)
        hospital_indices[hospital_names[3]] = idx_D
        remaining_indices = np.setdiff1d(remaining_indices, idx_D)
        
        # Hospital E: Mixed population
        hospital_indices[hospital_names[4]] = remaining_indices
        
        for name in hospital_names:
            hospital_datasets[name] = df.loc[hospital_indices[name]].reset_index(drop=True)
            
    else:
        raise ValueError(f"Unknown partition_type '{partition_type}'. Supported: 'iid', 'non_iid', 'dirichlet', 'demographic'.")
        
    return hospital_datasets

def save_hospital_partitions(config_path: str = "config/config.yaml") -> Dict[str, str]:
    config = load_config(config_path)
        
    raw_path = config["paths"]["raw_data_path"]
    hospital_dir = config["paths"]["hospital_data_dir"]
    results_dir = config["paths"].get("results_dir", "results")
    hospital_names = config["dataset"]["hospital_names"]
    partition_type = config["dataset"]["partition_type"]
    dirichlet_alpha = config["dataset"]["dirichlet_alpha"]
    random_seed = config["dataset"]["random_seed"]
    
    if not os.path.exists(raw_path):
        from data.generate_data import save_raw_dataset
        save_raw_dataset(config_path)
        
    df = pd.read_csv(raw_path)
    partitions = partition_dataset(
        df, 
        hospital_names=hospital_names,
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha,
        random_seed=random_seed
    )
    
    os.makedirs(hospital_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    saved_paths = {}
    
    stats_rows = []

    print(f"\n[PARTITIONER] Partitioning dataset into '{partition_type}' distribution for {len(hospital_names)} hospitals:")
    for name, hosp_df in partitions.items():
        clean_name = name.lower().replace(" ", "_")
        file_path = os.path.join(hospital_dir, f"{clean_name}.csv")
        hosp_df.to_csv(file_path, index=False)
        saved_paths[name] = file_path
        
        total_cases = len(hosp_df)
        pos_cases = int(hosp_df["disease_risk"].sum())
        neg_cases = total_cases - pos_cases
        prev = (pos_cases / total_cases * 100.0) if total_cases > 0 else 0.0
        avg_age = hosp_df["age"].mean()
        avg_bmi = hosp_df["bmi"].mean()
        avg_bp = hosp_df["blood_pressure_sys"].mean()
        avg_glucose = hosp_df["glucose"].mean()
        
        print(f"  • {name}: {total_cases} records | Disease Prevalence: {prev:.1f}% ({pos_cases}/{total_cases}) | Avg Age: {avg_age:.1f} | Avg BMI: {avg_bmi:.1f}")
        
        stats_rows.append({
            "hospital_name": name,
            "patient_count": total_cases,
            "positive_cases": pos_cases,
            "negative_cases": neg_cases,
            "disease_prevalence_pct": round(prev, 2),
            "avg_age": round(avg_age, 2),
            "avg_bmi": round(avg_bmi, 2),
            "avg_blood_pressure_sys": round(avg_bp, 2),
            "avg_glucose": round(avg_glucose, 2)
        })

    # Save results/hospital_distribution.csv
    dist_csv_path = os.path.join(results_dir, "hospital_distribution.csv")
    pd.DataFrame(stats_rows).to_csv(dist_csv_path, index=False)
    print(f"  • Saved hospital distribution summary to '{dist_csv_path}'.")

    return saved_paths

if __name__ == "__main__":
    save_hospital_partitions()
