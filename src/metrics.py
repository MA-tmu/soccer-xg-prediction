# src/metrics.py

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_clip_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)
    y_prob = _safe_clip_prob(y_prob)

    y_pred = (y_prob >= float(threshold)).astype(np.int32)

    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))

    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = 0.0

    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = 0.0

    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["ap"] = 0.0

    out["brier"] = float(brier_score_loss(y_true, y_prob))

    try:
        out["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        out["logloss"] = 0.0

    out["mae"] = float(np.mean(np.abs(y_prob - y_true.astype(np.float64))))
    return out

