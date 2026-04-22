"""
Метрики для оценки атак и защит.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)


def clean_accuracy(model, X_test, y_test) -> float:
    """Точность на чистом тесте (полезность модели)."""
    return accuracy_score(y_test, model.predict(X_test))


def attack_success_rate(model, X_attack, y_target) -> float:
    """Доля объектов, классифицированных как ЦЕЛЕВОЙ класс y_target.

    Для targeted/backdoor атак y_target — скалярная цель или
    вектор целей, одинаковой длины с X_attack.
    """
    y_pred = model.predict(X_attack)
    y_target = np.asarray(y_target)
    if y_target.ndim == 0:
        return float(np.mean(y_pred == int(y_target)))
    return float(np.mean(y_pred == y_target))


def robustness_score(clean_acc_before: float, clean_acc_after: float,
                     asr_before: float, asr_after: float) -> dict:
    """Комплексная метрика устойчивости.

    - acc_drop: падение точности от атаки без защиты;
    - acc_recovered: насколько защита восстановила точность;
    - asr_reduction: насколько защита снизила ASR.
    """
    return {
        "acc_drop": clean_acc_before - clean_acc_after,
        "acc_recovered": clean_acc_after,
        "asr_reduction": asr_before - asr_after,
    }


def detection_metrics(is_poison_true: np.ndarray,
                      is_poison_pred: np.ndarray) -> dict:
    """Метрики качества детекции отравленных объектов.

    is_poison_true — истинная маска отравлений;
    is_poison_pred — маска, которую защита пометила как отравление
                     (= NOT keep_mask_).
    """
    y_true = np.asarray(is_poison_true).astype(int)
    y_pred = np.asarray(is_poison_pred).astype(int)
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            out["roc_auc"] = float("nan")
    return out
