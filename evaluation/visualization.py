import os
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True

    # Global Matplotlib Typography & Sizing Configuration for High-DPI Dashboard Rendering
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.autolayout': True,
    })
except ImportError:
    PLOTS_AVAILABLE = False


def plot_training_curves(
    fl_history_no_dp: List[Dict[str, Any]], 
    fl_history_dp: Optional[List[Dict[str, Any]]] = None,
    save_path: str = "figures/training_convergence.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        print(f"[VISUALIZATION] matplotlib/seaborn not found; skipping image save for {save_path}.")
        return save_path

    rounds = [r["round"] for r in fl_history_no_dp]
    acc_no_dp = [r["accuracy"] * 100.0 for r in fl_history_no_dp]
    loss_no_dp = [r["loss"] for r in fl_history_no_dp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(rounds, acc_no_dp, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Federated (Standard FedAvg)")
    if fl_history_dp:
        acc_dp = [r["accuracy"] * 100.0 for r in fl_history_dp]
        ax1.plot(rounds, acc_dp, "s--", color="#d62728", linewidth=2.5, markersize=8, label="Federated + DP Noise (C=1.0, σ=0.1)")
    
    ax1.set_title("Global Model Accuracy vs. Communication Rounds", fontweight="bold", fontsize=15, pad=12)
    ax1.set_xlabel("Federated Communication Round", fontsize=13, labelpad=8)
    ax1.set_ylabel("Global Accuracy (%)", fontsize=13, labelpad=8)
    ax1.set_ylim([30, 100])
    ax1.legend(loc="lower right", fontsize=12, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(rounds, loss_no_dp, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Federated (Standard FedAvg)")
    if fl_history_dp:
        loss_dp = [r["loss"] for r in fl_history_dp]
        ax2.plot(rounds, loss_dp, "s--", color="#d62728", linewidth=2.5, markersize=8, label="Federated + DP Noise (C=1.0, σ=0.1)")
        
    ax2.set_title("Global Model Loss vs. Communication Rounds", fontweight="bold", fontsize=15, pad=12)
    ax2.set_xlabel("Federated Communication Round", fontsize=13, labelpad=8)
    ax2.set_ylabel("Loss", fontsize=13, labelpad=8)
    ax2.legend(loc="upper right", fontsize=12, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_roc_curves(
    method_results: Dict[str, Dict[str, Any]],
    save_path: str = "figures/roc_curve.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        return save_path

    plt.figure(figsize=(8, 6.5))

    colors = {
        "Local-Only": "#ff7f0e",
        "Centralized Baseline": "#2ca02c",
        "Federated (FedAvg)": "#1f77b4",
        "Federated + DP": "#d62728",
        "Federated + DP Noise + L2 Clip": "#d62728"
    }

    for method_name, res in method_results.items():
        if "roc_curve" in res and res["roc_curve"]:
            fpr = res["roc_curve"]["fpr"]
            tpr = res["roc_curve"]["tpr"]
            auc = res.get("roc_auc", 0.5)
            color = colors.get(method_name, "#333333")
            plt.plot(fpr, tpr, label=f"{method_name} (AUC = {auc:.3f})", color=color, linewidth=2.5)

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=13, labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", fontsize=15, pad=12)
    plt.legend(loc="lower right", fontsize=12, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_confusion_matrices(
    method_results: Dict[str, Dict[str, Any]],
    save_path: str = "figures/confusion_matrix.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        return save_path

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    methods = list(method_results.items())[:4]
    for idx, (method_name, res) in enumerate(methods):
        ax = axes[idx]
        raw_cm = res.get("confusion_matrix", [[0, 0], [0, 0]])
        cm = np.array(raw_cm, dtype=int)
        
        if cm.shape != (2, 2):
            padded_cm = np.zeros((2, 2), dtype=int)
            if cm.ndim == 2:
                r, c = min(2, cm.shape[0]), min(2, cm.shape[1])
                padded_cm[:r, :c] = cm[:r, :c]
            cm = padded_cm
        
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Low Risk (0)", "High Risk (1)"],
            yticklabels=["Low Risk (0)", "High Risk (1)"]
        )
        ax.set_title(f"{method_name}\nAcc: {res.get('accuracy', 0)*100:.1f}% | F1: {res.get('f1', 0):.3f}", fontweight="bold", fontsize=14, pad=10)
        ax.set_ylabel("True Ground Truth", fontsize=12)
        ax.set_xlabel("Predicted Disease Risk", fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_model_comparison_bar(
    method_results: Dict[str, Dict[str, Any]],
    save_path: str = "figures/model_comparison.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        return save_path

    methods = list(method_results.keys())
    accuracies = [method_results[m]["accuracy"] * 100.0 for m in methods]
    f1s = [method_results[m]["f1"] * 100.0 for m in methods]
    aucs = [method_results[m]["roc_auc"] * 100.0 for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='#1f77b4')
    rects2 = ax.bar(x, f1s, width, label='F1-Score (%)', color='#2ca02c')
    rects3 = ax.bar(x + width, aucs, width, label='ROC-AUC (%)', color='#ff7f0e')

    ax.set_ylabel('Performance (%)', fontsize=13, labelpad=8)
    ax.set_title('Comparative Benchmark Across Machine Learning Paradigms', fontweight='bold', fontsize=16, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10, ha='right', fontsize=12, fontweight='bold')
    ax.set_ylim([40, 105])
    ax.legend(loc='lower right', fontsize=12, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_hospital_demographics(
    hospital_data_dict: Dict[str, Any],
    save_path: str = "figures/hospital_distribution.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = list(hospital_data_dict.keys())
    counts = [len(df) for df in hospital_data_dict.values()]
    prevalences = [(df["disease_risk"].sum() / len(df)) * 100.0 for df in hospital_data_dict.values()]
    avg_ages = [df["age"].mean() for df in hospital_data_dict.values()]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, prevalences, width, label="Disease Prevalence (%)", color="#1f77b4")
    rects2 = ax1.bar(x + width/2, avg_ages, width, label="Average Patient Age", color="#ff7f0e")

    ax1.set_title("Hospital Demographics & Prevalence Skew (Non-IID)", fontweight="bold", fontsize=15, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=12)
    ax1.legend(loc="upper left", fontsize=11, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(names, counts, color="#2ca02c", alpha=0.85)
    ax2.set_title("Local Hospital Dataset Sizes (Patient Records)", fontweight="bold", fontsize=15, pad=12)
    ax2.set_ylabel("Sample Count", fontsize=13)
    ax2.set_xticklabels(names, fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_privacy_utility_tradeoff(
    privacy_results: List[Dict[str, Any]],
    save_path: str = "figures/privacy_utility_tradeoff.png"
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not PLOTS_AVAILABLE:
        return save_path

    plt.figure(figsize=(8, 5))

    sigmas = [r["noise_multiplier"] for r in privacy_results]
    accuracies = [r["accuracy"] * 100.0 for r in privacy_results]
    f1_scores = [r["f1"] for r in privacy_results]
    epsilons = [r.get("estimated_epsilon", "inf") for r in privacy_results]

    plt.plot(sigmas, accuracies, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Accuracy (%)")
    plt.plot(sigmas, [f * 100.0 for f in f1_scores], "s--", color="#2ca02c", linewidth=2.5, markersize=8, label="F1-Score (x100)")

    for i, txt in enumerate(epsilons):
        label_text = f"ε ≈ {txt}" if txt != float('inf') else "No DP"
        plt.annotate(
            label_text, 
            (sigmas[i], accuracies[i]), 
            textcoords="offset points", 
            xytext=(0, 10), 
            ha="center", 
            fontsize=10, 
            fontweight="bold"
        )

    plt.title("Privacy-Utility Trade-off (Gaussian Noise Multiplier vs Performance)", fontweight="bold", fontsize=15, pad=12)
    plt.xlabel("DP Noise Multiplier (sigma)", fontsize=13, labelpad=8)
    plt.ylabel("Performance Metric (%)", fontsize=13, labelpad=8)
    plt.ylim([30, 100])
    plt.legend(loc="lower left", fontsize=12, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path
