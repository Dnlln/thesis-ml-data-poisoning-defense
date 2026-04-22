"""
Модуль защит от атак отравления данных.

Реализованы следующие защитные механизмы:

- AnomalyDetectionDefense — детекция аномалий в признаковом
  пространстве методами Isolation Forest / LOF / Elliptic Envelope
  (на выбор пользователя);
- RONIDefense              — Reject On Negative Impact: отсев
  объектов, которые при добавлении в обучение УХУДШАЮТ качество
  на отложенной валидации;
- TrimmedLossDefense       — итеративное робастное обучение,
  отбрасывающее на каждой итерации top-α% объектов с наибольшей
  обучающей потерей (по аналогии с Trimmed Mean / Least Trimmed
  Squares);
- HybridPoisoningDefense   — АВТОРСКАЯ гибридная методика,
  последовательно применяющая три этапа:
  (1) AnomalyDetectionDefense  →  (2) RONIDefense  →
  (3) TrimmedLossDefense. Эта методика и является центральной
  в настоящей ВКР.

Все защиты имеют fit/predict интерфейс, совместимый с scikit-learn,
и дополнительно метод `sanitize(X, y)`, возвращающий индексы
принятых объектов (используется для метрик детекции).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor


# ---------------------------------------------------------------------------
# Базовый класс защиты
# ---------------------------------------------------------------------------

class BaseDefense(BaseEstimator, ClassifierMixin):
    """Базовый класс защиты. Подкласс обязан переопределить fit."""

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def sanitize(self, X, y) -> np.ndarray:
        """Возвращает булев массив: True, если объект признан чистым.

        По умолчанию все объекты считаются чистыми (заглушка).
        """
        return np.ones(len(X), dtype=bool)


# ---------------------------------------------------------------------------
# 1. Детекция аномалий
# ---------------------------------------------------------------------------

class AnomalyDetectionDefense(BaseDefense):
    """Защита на основе детекции аномалий.

    Идея: отравлённые объекты чаще всего лежат в "подозрительных"
    участках признакового пространства относительно других объектов
    своего класса. Для каждого класса обучаем один детектор
    аномалий (Isolation Forest / LOF / Elliptic Envelope) и
    отбрасываем объекты, получившие "аномальный" балл.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Базовая модель, обучаемая на очищенных данных.
    method : {"isoforest", "lof", "elliptic"}
        Метод детекции аномалий.
    contamination : float
        Ожидаемая доля аномалий (≈ ожидаемый poison_rate).
    per_class : bool
        Если True, детектор обучается отдельно для каждого класса
        (это обычно существенно повышает качество).
    """

    def __init__(self, base_estimator=None,
                 method: str = "isoforest",
                 contamination: float = 0.1,
                 per_class: bool = True,
                 random_state: int = 42):
        super().__init__(base_estimator)
        self.method = method
        self.contamination = contamination
        self.per_class = per_class
        self.random_state = random_state

    def _make_detector(self):
        if self.method == "isoforest":
            return IsolationForest(contamination=self.contamination,
                                   random_state=self.random_state)
        elif self.method == "lof":
            return LocalOutlierFactor(contamination=self.contamination,
                                      novelty=False)
        elif self.method == "elliptic":
            return EllipticEnvelope(contamination=self.contamination,
                                    support_fraction=1.0,
                                    random_state=self.random_state)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

    def sanitize(self, X, y) -> np.ndarray:
        """Возвращает маску "принятых" объектов (True = чистый)."""
        keep = np.ones(len(X), dtype=bool)
        X = np.asarray(X); y = np.asarray(y)
        if self.per_class:
            for c in np.unique(y):
                mask = y == c
                if mask.sum() < 10:
                    continue  # слишком мало данных для LOF/IF
                det = self._make_detector()
                if self.method == "lof":
                    pred = det.fit_predict(X[mask])
                else:
                    det.fit(X[mask])
                    pred = det.predict(X[mask])
                keep[mask] = pred == 1  # 1 — inlier, -1 — аномалия
        else:
            det = self._make_detector()
            if self.method == "lof":
                pred = det.fit_predict(X)
            else:
                det.fit(X)
                pred = det.predict(X)
            keep = pred == 1
        return keep

    def fit(self, X, y):
        keep = self.sanitize(X, y)
        self.keep_mask_ = keep
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        self.model_ = clone(base).fit(X[keep], y[keep])
        return self


# ---------------------------------------------------------------------------
# 2. RONI — Reject On Negative Impact
# ---------------------------------------------------------------------------

class RONIDefense(BaseDefense):
    """Reject On Negative Impact (Nelson et al., 2008).

    Для каждого объекта обучающей выборки (или батча объектов)
    проверяется: добавление его в обучающий набор УЛУЧШАЕТ или
    УХУДШАЕТ качество на отдельной валидационной выборке?
    Объекты, ухудшающие качество (negative impact), отбрасываются.

    Оригинальная RONI проверяет по одному объекту — это O(N)
    переобучений. Для ускорения реализована батчевая версия:
    объекты разбиваются на группы фиксированного размера `batch_size`,
    и отбрасывается вся группа, если она ухудшает валидацию.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Базовая модель.
    val_fraction : float
        Доля от X,y, используемая как валидация.
    batch_size : int
        Размер батча для батчевой RONI. При batch_size=1 получаем
        оригинальную однопроходную RONI.
    threshold : float
        Допустимое ухудшение accuracy (по умолчанию 0.0: отбрасываем
        всё, что ухудшает хоть на сколько-нибудь).
    """

    def __init__(self, base_estimator=None,
                 val_fraction: float = 0.25,
                 batch_size: int = 10,
                 threshold: float = 0.0,
                 random_state: int = 42):
        super().__init__(base_estimator)
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.threshold = threshold
        self.random_state = random_state

    def _score(self, est, X_val, y_val):
        return accuracy_score(y_val, est.predict(X_val))

    def sanitize(self, X, y) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X); y = np.asarray(y)
        n = len(X)

        # Стратифицированная валидация: берём val_fraction объектов
        # равномерно по классам.
        val_idx = []
        for c in np.unique(y):
            cls_idx = np.where(y == c)[0]
            n_val = max(2, int(round(self.val_fraction * len(cls_idx))))
            val_idx.extend(rng.choice(cls_idx, size=n_val, replace=False))
        val_idx = np.array(val_idx)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        X_val, y_val = X[val_idx], y[val_idx]
        X_tr0, y_tr0 = X[train_mask], y[train_mask]

        base = self.base_estimator or LogisticRegression(max_iter=1000)
        est_base = clone(base).fit(X_tr0, y_tr0)
        base_score = self._score(est_base, X_val, y_val)

        # Разбиваем кандидатов на батчи
        cand_idx = np.where(train_mask)[0]
        rng.shuffle(cand_idx)
        keep = np.ones(n, dtype=bool)
        keep[val_idx] = True  # валидацию всегда оставляем

        # Псевдо-начальный набор: 30% случайных объектов трейна
        init_size = max(10, int(0.3 * len(cand_idx)))
        init_idx = cand_idx[:init_size]
        remaining = cand_idx[init_size:]

        running_idx = list(init_idx)

        for start in range(0, len(remaining), self.batch_size):
            batch = remaining[start:start + self.batch_size]
            trial_idx = running_idx + list(batch)
            est_try = clone(base).fit(X[trial_idx], y[trial_idx])
            score = self._score(est_try, X_val, y_val)
            if score >= base_score - self.threshold:
                running_idx = trial_idx
                base_score = max(base_score, score)
            else:
                # negative impact: батч отбрасываем
                keep[batch] = False

        return keep

    def fit(self, X, y):
        keep = self.sanitize(X, y)
        self.keep_mask_ = keep
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        self.model_ = clone(base).fit(X[keep], y[keep])
        return self


# ---------------------------------------------------------------------------
# 3. Trimmed-Loss обучение
# ---------------------------------------------------------------------------

class TrimmedLossDefense(BaseDefense):
    """Робастное обучение с обрезкой максимальных потерь.

    На каждой итерации обучаем модель, вычисляем потери для всех
    объектов и выбрасываем top-α% объектов с наибольшей потерей.
    Это устойчивая к выбросам версия ERM (analog of Least Trimmed
    Squares для регрессии), описанная Jagielski et al. (2018)
    как "TRIM defense".

    Parameters
    ----------
    base_estimator : sklearn estimator
        Базовая модель.
    trim_ratio : float
        Доля объектов с наибольшей потерей, отбрасываемых на каждой
        итерации (≈ ожидаемый poison_rate).
    n_iter : int
        Число итераций trim-re-fit.
    """

    def __init__(self, base_estimator=None,
                 trim_ratio: float = 0.1,
                 n_iter: int = 5):
        super().__init__(base_estimator)
        self.trim_ratio = trim_ratio
        self.n_iter = n_iter

    def _losses(self, est, X, y):
        # log-loss на объект, с safe clip
        proba = est.predict_proba(X)
        # индексы истинных классов
        classes = list(est.classes_)
        yi = np.array([classes.index(v) for v in y])
        p_true = proba[np.arange(len(y)), yi]
        return -np.log(np.clip(p_true, 1e-12, 1.0))

    def sanitize(self, X, y) -> np.ndarray:
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        X = np.asarray(X); y = np.asarray(y)
        n = len(X)
        keep = np.ones(n, dtype=bool)

        for _ in range(self.n_iter):
            est = clone(base).fit(X[keep], y[keep])
            losses = self._losses(est, X, y)
            # Берём (1 - trim_ratio) с наименьшей потерей
            n_keep = int(round((1 - self.trim_ratio) * n))
            order = np.argsort(losses)
            keep = np.zeros(n, dtype=bool)
            keep[order[:n_keep]] = True
        return keep

    def fit(self, X, y):
        keep = self.sanitize(X, y)
        self.keep_mask_ = keep
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        self.model_ = clone(base).fit(X[keep], y[keep])
        return self


# ---------------------------------------------------------------------------
# 4. Гибридная авторская методика
# ---------------------------------------------------------------------------

class HybridPoisoningDefense(BaseDefense):
    """Авторская гибридная методика защиты (центральный вклад ВКР).

    Трёхступенчатая схема очистки обучающей выборки:

        Этап 1. Детекция аномалий (Isolation Forest, per-class).
                Отсеваем очевидные геометрические выбросы
                в пространстве признаков.

        Этап 2. Reject On Negative Impact (батчевая версия).
                Отсеваем подозрительные батчи, ухудшающие валидацию,
                — улавливает "хитрые" атаки вроде clean-label
                feature collision, не обязательно дающих аномальный
                балл, но искажающих решающую границу.

        Этап 3. Trimmed-Loss обучение на оставшихся данных.
                Финальная "полировка" — итеративная робастная ERM,
                окончательно снимающая оставшиеся отравлённые
                объекты с аномально большой потерей.

    Такая декомпозиция защищает от разных классов атак:
      • от label flipping и простых случайных инъекций — на Этапе 1;
      • от clean-label коллизий — на Этапе 2;
      • от "хвостовых" шумов и остатков — на Этапе 3.
    """

    def __init__(self, base_estimator=None,
                 contamination: float = 0.1,
                 use_anomaly: bool = True,
                 use_roni: bool = True,
                 use_trimmed_loss: bool = True,
                 anomaly_method: str = "isoforest",
                 roni_batch_size: int = 10,
                 trim_ratio: Optional[float] = None,
                 trim_n_iter: int = 3,
                 random_state: int = 42):
        super().__init__(base_estimator)
        self.contamination = contamination
        self.use_anomaly = use_anomaly
        self.use_roni = use_roni
        self.use_trimmed_loss = use_trimmed_loss
        self.anomaly_method = anomaly_method
        self.roni_batch_size = roni_batch_size
        self.trim_ratio = trim_ratio
        self.trim_n_iter = trim_n_iter
        self.random_state = random_state

    def sanitize(self, X, y) -> np.ndarray:
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        X = np.asarray(X); y = np.asarray(y)
        n = len(X)
        keep = np.ones(n, dtype=bool)

        # Этап 1
        if self.use_anomaly:
            ad = AnomalyDetectionDefense(
                base_estimator=base,
                method=self.anomaly_method,
                contamination=self.contamination,
                random_state=self.random_state,
            )
            keep &= ad.sanitize(X, y)

        # Этап 2 (RONI применяется к тому, что осталось)
        if self.use_roni and keep.sum() > 20:
            roni = RONIDefense(
                base_estimator=base,
                val_fraction=0.25,
                batch_size=self.roni_batch_size,
                random_state=self.random_state,
            )
            keep_sub = roni.sanitize(X[keep], y[keep])
            # Восстанавливаем полноразмерную маску
            idx = np.where(keep)[0]
            new_keep = np.zeros(n, dtype=bool)
            new_keep[idx[keep_sub]] = True
            keep = new_keep

        # Этап 3
        if self.use_trimmed_loss and keep.sum() > 20:
            tr_ratio = self.trim_ratio
            if tr_ratio is None:
                # Оценка остаточной контаминации:
                # предполагаем, что часть яда пережила первые два этапа.
                tr_ratio = max(0.02, self.contamination * 0.3)
            tl = TrimmedLossDefense(
                base_estimator=base,
                trim_ratio=tr_ratio,
                n_iter=self.trim_n_iter,
            )
            keep_sub = tl.sanitize(X[keep], y[keep])
            idx = np.where(keep)[0]
            new_keep = np.zeros(n, dtype=bool)
            new_keep[idx[keep_sub]] = True
            keep = new_keep

        return keep

    def fit(self, X, y):
        keep = self.sanitize(X, y)
        self.keep_mask_ = keep
        base = self.base_estimator or LogisticRegression(max_iter=1000)
        self.model_ = clone(base).fit(X[keep], y[keep])
        return self
