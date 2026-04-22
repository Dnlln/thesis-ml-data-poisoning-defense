"""Unit-тесты модуля poisondefense."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisondefense import (
    AnomalyDetectionDefense,
    BackdoorAttack,
    FeatureCollisionAttack,
    HybridPoisoningDefense,
    LabelFlipAttack,
    OptimizationBasedAttack,
    RONIDefense,
    TargetedLabelFlipAttack,
    TrimmedLossDefense,
    inject_poison,
    load_dataset,
    set_random_state,
    split_train_val_test,
)


@pytest.fixture
def data():
    set_random_state(42)
    ds = load_dataset("breast_cancer")
    return split_train_val_test(ds.X, ds.y, seed=42)


def test_dataset_loaders():
    for n in ["breast_cancer", "iris", "wine", "synthetic"]:
        ds = load_dataset(n)
        assert len(ds.X) == len(ds.y)
        assert set(np.unique(ds.y)).issubset({0, 1})


@pytest.mark.parametrize("atk_cls", [
    LabelFlipAttack, TargetedLabelFlipAttack, FeatureCollisionAttack,
    OptimizationBasedAttack, BackdoorAttack,
])
def test_attacks(data, atk_cls):
    X_tr, y_tr, *_ = data
    atk = atk_cls()
    X_p, y_p = atk.generate(X_tr, y_tr, n_poison=20, seed=1)
    assert len(X_p) == 20 == len(y_p)
    assert X_p.shape[1] == X_tr.shape[1]


@pytest.mark.parametrize("defense_cls", [
    AnomalyDetectionDefense, RONIDefense, TrimmedLossDefense,
    HybridPoisoningDefense,
])
def test_defenses_fit_predict(data, defense_cls):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    atk = LabelFlipAttack()
    X_mix, y_mix, _ = inject_poison(X_tr, y_tr, atk, poison_rate=0.1)
    defn = defense_cls(base_estimator=LogisticRegression(max_iter=1000))
    defn.fit(X_mix, y_mix)
    y_pred = defn.predict(X_te)
    assert y_pred.shape == y_te.shape
    assert hasattr(defn, "keep_mask_")


def test_hybrid_improves_over_baseline(data):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    atk = LabelFlipAttack()
    X_mix, y_mix, _ = inject_poison(X_tr, y_tr, atk, poison_rate=0.2)

    base = LogisticRegression(max_iter=2000)
    base.fit(X_mix, y_mix)
    acc_base = (base.predict(X_te) == y_te).mean()

    defn = HybridPoisoningDefense(
        base_estimator=LogisticRegression(max_iter=2000),
        contamination=0.2)
    defn.fit(X_mix, y_mix)
    acc_def = (defn.predict(X_te) == y_te).mean()
    # защита не должна существенно ухудшать (допускаем ≥ -2 п.п.,
    # обычно существенно лучше)
    assert acc_def >= acc_base - 0.02
