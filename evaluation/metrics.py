from typing import Dict, Any, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score as sk_acc, precision_score as sk_prec, recall_score as sk_rec, f1_score as sk_f1,
        roc_auc_score as sk_auc, confusion_matrix as sk_cm, roc_curve as sk_roc
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds decision threshold between 0.1 and 0.9 that maximizes F1 score."""
    best_th = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.10, 0.90, 81)
    for th in thresholds:
        pred_th = (y_prob >= th).astype(int)
        if SKLEARN_AVAILABLE:
            score = float(sk_f1(y_true, pred_th, zero_division=0))
        else:
            tp = np.sum((y_true == 1) & (pred_th == 1))
            fp = np.sum((y_true == 0) & (pred_th == 1))
            fn = np.sum((y_true == 1) & (pred_th == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return best_th

def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray,
    optimize_threshold: bool = True
) -> Dict[str, Any]:
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    
    if optimize_threshold and len(np.unique(y_true)) > 1:
        best_th = find_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= best_th).astype(int)
    else:
        y_pred = np.array(y_pred).astype(int)

    if SKLEARN_AVAILABLE:
        acc = float(sk_acc(y_true, y_pred))
        prec = float(sk_prec(y_true, y_pred, zero_division=0))
        rec = float(sk_rec(y_true, y_pred, zero_division=0))
        f1 = float(sk_f1(y_true, y_pred, zero_division=0))
        try:
            auc = float(sk_auc(y_true, y_prob))
        except ValueError:
            auc = 0.5
        cm = sk_cm(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr, tpr, thresholds = sk_roc(y_true, y_prob)
    else:
        # Pure NumPy metric calculations
        acc = float(np.mean(y_true == y_pred))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Empirical ROC curve calculation
        thresholds = np.linspace(0, 1, 50)
        fpr_list = []
        tpr_list = []
        pos_total = max(1, np.sum(y_true == 1))
        neg_total = max(1, np.sum(y_true == 0))

        for th in thresholds:
            p_th = (y_prob >= th).astype(int)
            tpr_val = np.sum((y_true == 1) & (p_th == 1)) / pos_total
            fpr_val = np.sum((y_true == 0) & (p_th == 1)) / neg_total
            tpr_list.append(tpr_val)
            fpr_list.append(fpr_val)

        # Sort for trapezoidal integration
        order = np.argsort(fpr_list)
        fpr = np.array(fpr_list)[order]
        tpr = np.array(tpr_list)[order]
        auc = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else 0.5
        cm = np.array([[tn, fp], [fn, tp]])

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    }

