import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, y_proba):
    """Порог, при котором F1 максимален."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # 1e-9 защищает от деления на ноль, последняя точка кривой без порога
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_proba)
    return {
        "roc_auc": roc_auc,
        "gini": 2 * roc_auc - 1,
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
