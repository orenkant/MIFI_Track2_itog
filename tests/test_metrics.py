import numpy as np
import pytest

from src.models.metrics import compute_metrics, find_best_threshold


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["gini"] == 1.0
    assert m["precision"] == m["recall"] == m["f1"] == 1.0


def test_threshold_changes_class_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.45, 0.9])
    low = compute_metrics(y, proba, threshold=0.3)
    high = compute_metrics(y, proba, threshold=0.5)
    assert low["recall"] == 1.0
    assert high["recall"] == 0.5
    # ROC-AUC от порога не зависит
    assert low["roc_auc"] == high["roc_auc"]


def test_no_positive_predictions_does_not_fail():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_metrics(y, proba, threshold=0.9)
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0


def test_metric_keys():
    m = compute_metrics(np.array([0, 1]), np.array([0.3, 0.7]))
    expected = {"roc_auc", "gini", "pr_auc", "precision", "recall", "f1", "accuracy"}
    assert set(m) == expected


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    t = find_best_threshold(y, proba)
    assert 0.3 < t <= 0.6
    assert compute_metrics(y, proba, t)["f1"] == pytest.approx(1.0)
