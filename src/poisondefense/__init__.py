"""
poisondefense — программный модуль защиты классических моделей
машинного обучения от атак отравления данных (data poisoning).

Модуль реализует авторскую гибридную методику защиты, основанную
на последовательном применении:

    1) детекции аномалий в признаковом пространстве и пространстве
       "признаки+метка" (Isolation Forest / LOF / Elliptic Envelope);
    2) фильтрации методом RONI (Reject On Negative Impact) —
       отсева объектов, ухудшающих качество на валидационной выборке;
    3) робастного обучения с итеративной обрезкой потерь
       (Trimmed-Loss training) и робастных оценок параметров.

Пример использования:

    >>> from poisondefense import HybridPoisoningDefense
    >>> from sklearn.linear_model import LogisticRegression
    >>> defense = HybridPoisoningDefense(
    ...     base_estimator=LogisticRegression(max_iter=1000),
    ...     contamination=0.1,
    ...     use_anomaly=True,
    ...     use_roni=True,
    ...     use_trimmed_loss=True,
    ... )
    >>> defense.fit(X_train, y_train)
    >>> y_pred = defense.predict(X_test)

Автор: ВКР по теме "Методика защиты моделей машинного обучения
от атак с использованием отравления данных".
"""

from .attacks import (
    LabelFlipAttack,
    TargetedLabelFlipAttack,
    FeatureCollisionAttack,
    OptimizationBasedAttack,
    BackdoorAttack,
)
from .defenses import (
    AnomalyDetectionDefense,
    RONIDefense,
    TrimmedLossDefense,
    HybridPoisoningDefense,
)
from .metrics import (
    attack_success_rate,
    clean_accuracy,
    robustness_score,
    detection_metrics,
)
from .utils import (
    load_dataset,
    split_train_val_test,
    inject_poison,
    set_random_state,
)

__version__ = "1.0.0"
__author__ = "ВКР 2026"

__all__ = [
    # attacks
    "LabelFlipAttack",
    "TargetedLabelFlipAttack",
    "FeatureCollisionAttack",
    "OptimizationBasedAttack",
    "BackdoorAttack",
    # defenses
    "AnomalyDetectionDefense",
    "RONIDefense",
    "TrimmedLossDefense",
    "HybridPoisoningDefense",
    # metrics
    "attack_success_rate",
    "clean_accuracy",
    "robustness_score",
    "detection_metrics",
    # utils
    "load_dataset",
    "split_train_val_test",
    "inject_poison",
    "set_random_state",
]
