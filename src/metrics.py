"""Evaluation metrics for KL grading."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    n = int(proba.shape[1]) if proba.ndim == 2 else int(np.max(y_true)) + 1
    if class_names is None:
        class_names = [f"KL{i}" for i in range(n)]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    prec = precision_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)

    auc = None
    if proba.shape[0] == len(y_true) and proba.shape[1] == n:
        # inputs may be logits; sklearn >=1.9 requires row-normalised probabilities
        e = np.exp(proba - proba.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        try:
            auc = {
                "macro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
                "micro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="micro")),
                "per_class": {
                    class_names[i]: float(
                        roc_auc_score((y_true == i).astype(int), probs[:, i])
                    )
                    for i in range(n)
                },
            }
        except ValueError:
            auc = None

    per_class = [
        {
            "class": class_names[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1c[i]),
            "auc_ovr": (auc["per_class"][class_names[i]] if auc else None),
        }
        for i in range(n)
    ]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "auc": auc,
    }
