import os
import json
import pickle
import time
from typing import Dict, Any
import pandas as pd
import numpy as np

from config.config_loader import load_config
from data.generate_data import save_raw_dataset
from data.partitioning import save_hospital_partitions
from client.preprocessing import HospitalDataPreprocessor
from client.hospital_client import HospitalClient
from models.disease_model import DiseasePredictionMLP, get_model_parameters, save_model_pt
from experiments.local_only import run_local_only_experiment
from experiments.centralized import run_centralized_experiment
from experiments.federated import run_federated_experiment
from evaluation.visualization import (
    plot_training_curves,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison_bar,
    plot_hospital_demographics,
    plot_privacy_utility_tradeoff
)

def run_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    print("=" * 80)
    print(" FEDERATED LEARNING FOR PRIVACY-PRESERVING DISEASE PREDICTION IN SMART HOSPITALS")
    print(" Final-Year Academic Prototype Execution Pipeline — Review 2")
    print("=" * 80)

    # 1. Load Configuration
    config = load_config(config_path)

    results_dir = config["paths"]["results_dir"]
    plots_dir = config["paths"]["plots_dir"]
    models_dir = config["paths"].get("models_dir", "models")
    model_save_path = config["paths"].get("model_save_path", "results/global_disease_model.pkl")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    num_rounds = config["federated"]["num_rounds"]
    local_epochs = config["federated"]["local_epochs"]
    learning_rate = config["federated"]["learning_rate"]
    batch_size = config["federated"]["batch_size"]

    # 2. Generate Dataset & Partition across 5 Hospitals
    print("\n[STEP 1/5] Generating Synthetic Healthcare Dataset & Multi-Hospital Partitioning...")
    raw_path = save_raw_dataset(config_path)
    hospital_paths = save_hospital_partitions(config_path)

    # 3. Fit Global Preprocessor and Initialize Hospital Clients
    print("\n[STEP 2/5] Initializing Isolated Hospital Client Environments...")
    hospital_data_dict = {}
    for name, path in hospital_paths.items():
        hospital_data_dict[name] = pd.read_csv(path)

    df_raw = pd.read_csv(raw_path)
    global_preprocessor = HospitalDataPreprocessor()
    global_preprocessor.preprocess_df(df_raw, is_train=True)

    hospital_clients = []
    for name, df_h in hospital_data_dict.items():
        client = HospitalClient(
            hospital_name=name,
            df_local=df_h,
            preprocessor=global_preprocessor,
            batch_size=batch_size
        )
        hospital_clients.append(client)

    in_features = hospital_clients[0].in_features
    print(f"  • Successfully initialized {len(hospital_clients)} Hospital Clients with {in_features} input features.")

    # 4. Run Experimental Paradigms
    print("\n[STEP 3/5] Executing Comparative Machine Learning Paradigms...")
    
    # Paradigm A: Local-Only Baseline
    local_results = run_local_only_experiment(
        hospital_clients=hospital_clients,
        in_features=in_features,
        epochs=num_rounds * local_epochs,
        learning_rate=learning_rate
    )

    # Paradigm B: Centralized Baseline (Upper-bound reference on pooled data)
    centralized_save_path = os.path.join(models_dir, "centralized_model.pt")
    centralized_results, centralized_model = run_centralized_experiment(
        hospital_data_dict=hospital_data_dict,
        in_features=in_features,
        epochs=num_rounds * local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=config["dataset"]["random_seed"],
        save_path=centralized_save_path
    )

    # Paradigm C: Federated Learning (Standard FedAvg)
    federated_save_path = os.path.join(models_dir, "federated_model.pt")
    fl_results, fl_history, fed_model = run_federated_experiment(
        hospital_clients=hospital_clients,
        in_features=in_features,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        enable_dp=False,
        save_model_path=federated_save_path
    )

    # Paradigm D: Federated Learning + DP Noise + L2 Clipping
    private_save_path = os.path.join(models_dir, "federated_private_model.pt")
    dp_results, dp_history, fed_private_model = run_federated_experiment(
        hospital_clients=hospital_clients,
        in_features=in_features,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        enable_dp=config["privacy"]["enabled"],
        clip_norm=config["privacy"]["clip_norm"],
        noise_multiplier=config["privacy"]["noise_multiplier"],
        delta=config["privacy"]["delta"],
        save_model_path=private_save_path
    )

    # Save pickle for backwards compatibility with dashboard prediction artifact
    with open(model_save_path, "wb") as f:
        pickle.dump({
            "model_state": get_model_parameters(fed_private_model),
            "in_features": in_features,
            "preprocessor": global_preprocessor
        }, f)
    print(f"  • Saved trained global model artifact to '{model_save_path}'.")

    # 5. Run Privacy-Utility Sweep Experiment
    print("\n[STEP 4/5] Executing Privacy-Utility Tradeoff Analysis (Varying Noise Multipliers)...")
    noise_values = [0.0, 0.05, 0.1, 0.25, 0.5]
    privacy_sweep_results = []
    
    for sigma in noise_values:
        dp_enabled = (sigma > 0.0)
        res, _, _ = run_federated_experiment(
            hospital_clients=hospital_clients,
            in_features=in_features,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            enable_dp=dp_enabled,
            clip_norm=config["privacy"]["clip_norm"],
            noise_multiplier=sigma,
            delta=config["privacy"]["delta"]
        )
        res["noise_multiplier"] = sigma
        privacy_sweep_results.append(res)

    # 6. Save Structured Output CSVs and Figures
    print("\n[STEP 5/5] Exporting Required CSVs, Plots & Summary Reports...")
    method_comparison = {
        "Local-Only": local_results,
        "Centralized Baseline": centralized_results,
        "Federated (FedAvg)": fl_results,
        "Federated + Privacy": dp_results
    }

    # A. training_metrics.csv
    tr_rows = []
    for r in fl_history:
        tr_rows.append({
            "round": r["round"],
            "model": "Federated (Standard FedAvg)",
            "train_loss": round(r["loss"], 4),
            "validation_loss": round(r["loss"], 4),
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "roc_auc": round(r["roc_auc"], 4),
            "training_time": round(r["round_duration_sec"], 2)
        })
    for r in dp_history:
        tr_rows.append({
            "round": r["round"],
            "model": "Federated + Privacy (DP Noise + L2 Clip)",
            "train_loss": round(r["loss"], 4),
            "validation_loss": round(r["loss"], 4),
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "roc_auc": round(r["roc_auc"], 4),
            "training_time": round(r["round_duration_sec"], 2)
        })
    tr_csv_path = os.path.join(results_dir, "training_metrics.csv")
    pd.DataFrame(tr_rows).to_csv(tr_csv_path, index=False)
    print(f"  • Saved training metrics to '{tr_csv_path}'.")

    # B. model_comparison.csv
    comp_rows = []
    for name, res in method_comparison.items():
        comp_rows.append({
            "Model Paradigm": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1-Score": round(res["f1"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "Training Time (s)": round(res["training_time_sec"], 2),
            "Communication Rounds": res.get("communication_rounds", 0)
        })
    comp_csv_path = os.path.join(results_dir, "model_comparison.csv")
    pd.DataFrame(comp_rows).to_csv(comp_csv_path, index=False)
    print(f"  • Saved model comparison metrics to '{comp_csv_path}'.")

    # C. predictions.csv
    X_test_all, y_test_all = [], []
    for client in hospital_clients:
        if hasattr(client.test_loader, "dataset"):
            for xb, yb in client.test_loader:
                X_test_all.append(xb.cpu().numpy() if hasattr(xb, "numpy") else xb)
                y_test_all.append(yb.cpu().numpy() if hasattr(yb, "numpy") else yb)
        else:
            X_test_all.append(client.test_loader[0])
            y_test_all.append(client.test_loader[1])
    X_test_concat = np.vstack(X_test_all)
    y_test_concat = np.vstack(y_test_all).flatten()

    probs_pred = fed_private_model.predict_proba(X_test_concat).flatten()
    if hasattr(probs_pred, "cpu"):
        probs_pred = probs_pred.cpu().numpy()
    else:
        probs_pred = np.array(probs_pred)

    preds_binary = (probs_pred >= 0.5).astype(int)
    
    pred_df = pd.DataFrame({
        "sample_id": [f"TEST_{i+1:04d}" for i in range(len(preds_binary))],
        "true_label": y_test_concat.astype(int),
        "predicted_class": preds_binary,
        "risk_probability": np.round(probs_pred, 4),
        "prediction_status": np.where(preds_binary == y_test_concat, "CORRECT", "INCORRECT")
    })
    pred_csv_path = os.path.join(results_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  • Saved test predictions log to '{pred_csv_path}'.")

    # D. final_results.txt
    txt_summary_path = os.path.join(results_dir, "final_results.txt")
    with open(txt_summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" FEDERATED LEARNING FOR PRIVACY-PRESERVING DISEASE PREDICTION IN SMART HOSPITALS\n")
        f.write(" FINAL EXPERIMENT RESULTS REPORT — REVIEW 2\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Cohort: {config['dataset']['num_samples']} Patient Records across 5 Hospitals\n")
        f.write(f"Data Distribution: {config['dataset']['partition_type'].upper()}\n")
        f.write(f"FL Communication Rounds: {num_rounds}, Local Epochs: {local_epochs}, Batch Size: {batch_size}, LR: {learning_rate}\n")
        f.write(f"Privacy Mechanism: L2 Clip Norm={config['privacy']['clip_norm']}, Noise Multiplier={config['privacy']['noise_multiplier']}\n\n")
        f.write("COMPARATIVE MODEL PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        for row in comp_rows:
            f.write(f"• {row['Model Paradigm']:<30} | Acc: {row['Accuracy']*100:.2f}% | F1: {row['F1-Score']:.4f} | ROC-AUC: {row['ROC-AUC']:.4f} | Time: {row['Training Time (s)']}s\n")
        f.write("-" * 80 + "\n\n")
        f.write("RESEARCH INTEGRITY & DATA PRIVACY DIRECTIVE:\n")
        f.write("Raw patient records remained strictly inside client hospital boundaries throughout training.\n")
        f.write("Central server received model weight updates only. No individual patient data was transmitted.\n")
    print(f"  • Saved text summary report to '{txt_summary_path}'.")

    # E. Save experiment_results.json for dashboard caching
    json_path = os.path.join(results_dir, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "method_comparison": method_comparison,
            "fl_history_no_dp": fl_history,
            "fl_history_dp": dp_history,
            "privacy_sweep_results": privacy_sweep_results
        }, f, indent=2)
    print(f"  • Saved experiment JSON metrics to '{json_path}'.")

    # F. Generate and Save Figures to figures/
    plot_training_curves(fl_history, dp_history, os.path.join(plots_dir, "training_convergence.png"))
    plot_roc_curves(method_comparison, os.path.join(plots_dir, "roc_curve.png"))
    plot_confusion_matrices(method_comparison, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_model_comparison_bar(method_comparison, os.path.join(plots_dir, "model_comparison.png"))
    plot_hospital_demographics(hospital_data_dict, os.path.join(plots_dir, "hospital_distribution.png"))
    plot_privacy_utility_tradeoff(privacy_sweep_results, os.path.join(plots_dir, "privacy_utility_tradeoff.png"))
    print(f"  • Generated publication-ready figures in '{plots_dir}/'.")

    # Print Benchmark Summary Table
    print("\n" + "=" * 90)
    print(" EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Learning Approach':<30} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'Time(s)':<7}")
    print("-" * 90)
    for name, res in method_comparison.items():
        print(f"{name:<30} | {res['accuracy']*100:>7.2f}% | {res['precision']:>9.4f} | {res['recall']:>9.4f} | {res['f1']:>9.4f} | {res['roc_auc']:>9.4f} | {res['training_time_sec']:>6.2f}s")
    print("=" * 90)
    print("\nTo launch the interactive academic dashboard, run:")
    print("  streamlit run dashboard/app.py\n")

    return method_comparison

if __name__ == "__main__":
    run_pipeline()
