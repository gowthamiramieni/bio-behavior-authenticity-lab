"""
Biometric authentication evaluation metrics:
FAR (False Acceptance Rate), FRR (False Rejection Rate),
EER (Equal Error Rate), ROC Curve, AUC, Confusion Matrix, and Classification Metrics.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) and the optimal decision threshold
    where FAR == FRR (or the point of intersection).

    Returns:
        eer (float): Equal Error Rate in [0, 1]
        eer_threshold (float): Decision threshold at EER
        fpr (np.ndarray): False Acceptance Rate / FPR across thresholds
        fnr (np.ndarray): False Rejection Rate / FNR across thresholds
        thresholds (np.ndarray): Decision thresholds
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    # Scikit-learn roc_curve:
    # fpr = FP / (FP + TN) -> FAR (impostors accepted as genuine)
    # tpr = TP / (TP + FN) -> 1 - FRR
    # fnr = 1 - tpr -> FRR (genuine rejected as impostor)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FAR and FRR are closest
    diff = np.abs(fpr - fnr)
    idx = np.nanargmin(diff)

    # Linear interpolation for refined EER estimate if between points
    if idx < len(fpr) - 1:
        x1, x2 = fpr[idx], fpr[idx + 1]
        y1, y2 = fnr[idx], fnr[idx + 1]
        denom = (y2 - y1) - (x2 - x1)
        if abs(denom) > 1e-7:
            alpha = (x1 - y1) / denom
            if 0 <= alpha <= 1:
                eer = float(x1 + alpha * (x2 - x1))
            else:
                eer = float((fpr[idx] + fnr[idx]) / 2.0)
        else:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

    eer_threshold = float(thresholds[idx])
    # Handle infinite threshold from sklearn roc_curve edge case
    if np.isinf(eer_threshold):
        eer_threshold = 1.0

    return eer, eer_threshold, fpr, fnr, thresholds


def compute_far_frr_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Computes FAR and FRR for a specific decision threshold.
    y_pred = 1 if score >= threshold else 0.
    FAR = False Positives / All Impostors (Negative instances)
    FRR = False Negatives / All Genuine (Positive instances)
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    # Positive = Genuine (1), Negative = Impostor (0)
    impostor_mask = (y_true == 0)
    genuine_mask = (y_true == 1)

    num_impostors = int(np.sum(impostor_mask))
    num_genuine = int(np.sum(genuine_mask))

    if num_impostors == 0:
        far = 0.0
    else:
        false_accepts = int(np.sum((y_pred == 1) & impostor_mask))
        far = float(false_accepts / num_impostors)

    if num_genuine == 0:
        frr = 0.0
    else:
        false_rejects = int(np.sum((y_pred == 0) & genuine_mask))
        frr = float(false_rejects / num_genuine)

    return far, frr


def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of biometric verification scores.
    Returns dictionary with all metrics, curves, and confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    # Check for empty or single-class array
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "eer": 0.5,
            "eer_threshold": 0.5,
            "auc": 0.5,
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "far": 0.5,
            "frr": 0.5,
            "confusion_matrix": [[0, 0], [0, 0]],
            "roc_fpr": [0.0, 1.0],
            "roc_tpr": [0.0, 1.0],
            "thresholds": [1.0, 0.0],
            "fnr": [1.0, 0.0],
        }

    # EER calculation
    eer, eer_thresh, fpr, fnr, threshs = compute_eer(y_true, y_scores)

    # ROC AUC
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    # Use EER threshold if none provided
    dec_thresh = eer_thresh if threshold is None else float(threshold)
    y_pred = (y_scores >= dec_thresh).astype(int)

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    far, frr = compute_far_frr_at_threshold(y_true, y_scores, dec_thresh)

    return {
        "eer": round(eer, 4),
        "eer_threshold": round(dec_thresh, 4),
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": [round(float(x), 4) for x in fpr],
        "roc_tpr": [round(float(1.0 - x), 4) for x in fnr],
        "fnr": [round(float(x), 4) for x in fnr],
        "thresholds": [round(float(x), 4) if np.isfinite(x) else 1.0 for x in threshs],
    }
