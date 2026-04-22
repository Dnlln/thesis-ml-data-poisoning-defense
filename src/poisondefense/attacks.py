"""
Модуль атак отравления данных (data poisoning) на классические
модели машинного обучения.

Реализованы следующие атаки:

- LabelFlipAttack          — случайное изменение меток (availability);
- TargetedLabelFlipAttack  — целевое изменение меток для выбранного
                              целевого класса;
- FeatureCollisionAttack   — "clean-label" атака: подбор точек,
                              лежащих рядом с точками целевого класса,
                              но с сохранённой (исходной) меткой;
- OptimizationBasedAttack  — градиентная атака на логистическую
                              регрессию, основанная на приближении
                              bilevel-оптимизации (back-gradient);
- BackdoorAttack           — простая бэкдор-атака с триггером
                              в виде фиксированного паттерна признаков.

Все атаки наследуются от BaseAttack и реализуют метод `generate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class BaseAttack(ABC):
    """Абстрактный класс атаки отравления."""

    name: str = "base"

    @abstractmethod
    def generate(self, X: np.ndarray, y: np.ndarray,
                 n_poison: int, seed: int = 42
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Генерирует набор отравлённых объектов.

        Parameters
        ----------
        X, y : np.ndarray
            Чистая обучающая выборка, относительно которой
            конструируется атака.
        n_poison : int
            Число отравлённых объектов.
        seed : int
            Seed генератора случайных чисел.

        Returns
        -------
        X_p, y_p : np.ndarray
            Отравлённые объекты и их (ложные) метки.
        """
        ...


# ---------------------------------------------------------------------------
# Label flipping
# ---------------------------------------------------------------------------

class LabelFlipAttack(BaseAttack):
    """Случайное переворачивание меток.

    Для произвольно выбранных объектов из обучающей выборки
    метка заменяется на случайную другую. Это классическая
    "availability"-атака: цель — снизить общее качество модели.
    """

    name = "label_flip"

    def __init__(self, subsample_from_clean: bool = True):
        self.subsample_from_clean = subsample_from_clean

    def generate(self, X, y, n_poison, seed=42):
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        if self.subsample_from_clean:
            idx = rng.choice(len(X), size=n_poison, replace=True)
            X_p = X[idx].copy()
            y_orig = y[idx].copy()
        else:
            # Генерация "из середины" — не требуется классически
            raise NotImplementedError
        # Переворачиваем метки на случайные отличные
        y_p = y_orig.copy()
        for i in range(len(y_p)):
            others = classes[classes != y_orig[i]]
            y_p[i] = rng.choice(others)
        return X_p, y_p


class TargetedLabelFlipAttack(BaseAttack):
    """Целенаправленное переворачивание меток.

    Атакующий выбирает объекты источникового класса `source_class`
    и помечает их меткой целевого класса `target_class`, чтобы
    сместить границу принятия решений в нужную сторону.
    Выбираются объекты, ближайшие (по евклидовой метрике) к классу
    target_class — они наиболее эффективны для сдвига границы.
    """

    name = "targeted_label_flip"

    def __init__(self, source_class: int = 0, target_class: int = 1,
                 strategy: str = "nearest"):
        self.source_class = source_class
        self.target_class = target_class
        assert strategy in ("nearest", "random")
        self.strategy = strategy

    def generate(self, X, y, n_poison, seed=42):
        rng = np.random.default_rng(seed)
        src_mask = y == self.source_class
        tgt_mask = y == self.target_class
        X_src = X[src_mask]
        X_tgt = X[tgt_mask]
        if len(X_src) == 0 or len(X_tgt) == 0:
            raise ValueError("В выборке отсутствуют объекты одного из классов.")

        if self.strategy == "random":
            idx = rng.choice(len(X_src), size=min(n_poison, len(X_src)),
                             replace=len(X_src) < n_poison)
        else:  # nearest
            nn = NearestNeighbors(n_neighbors=1).fit(X_tgt)
            dists, _ = nn.kneighbors(X_src)
            # Берём n_poison ближайших к target кластеру
            order = np.argsort(dists.ravel())
            idx = order[:n_poison]
            # Если недостаточно — расширяем случайно
            if len(idx) < n_poison:
                extra = rng.choice(len(X_src), size=n_poison - len(idx),
                                   replace=True)
                idx = np.concatenate([idx, extra])

        X_p = X_src[idx].copy()
        y_p = np.full(len(X_p), self.target_class, dtype=y.dtype)
        return X_p, y_p


# ---------------------------------------------------------------------------
# Feature collision (clean-label)
# ---------------------------------------------------------------------------

class FeatureCollisionAttack(BaseAttack):
    """Clean-label атака "коллизия признаков".

    Берём объект целевого класса t и двигаем его в признаковом
    пространстве в направлении кластера класса s, но сохраняем
    исходную метку t. Такие объекты выглядят "чистыми" (метка
    корректна с точки зрения простого эксперта), но разрушают
    геометрию обучающей выборки.
    """

    name = "feature_collision"

    def __init__(self, source_class: int = 0, target_class: int = 1,
                 alpha: float = 0.5):
        """
        alpha задаёт степень смещения: 0 — не смещать, 1 — на
        центроид противоположного класса.
        """
        self.source_class = source_class
        self.target_class = target_class
        self.alpha = float(alpha)

    def generate(self, X, y, n_poison, seed=42):
        rng = np.random.default_rng(seed)
        X_tgt = X[y == self.target_class]
        X_src = X[y == self.source_class]
        if len(X_src) == 0 or len(X_tgt) == 0:
            raise ValueError("Отсутствуют объекты одного из классов.")
        centroid_src = X_src.mean(axis=0)
        idx = rng.choice(len(X_tgt), size=n_poison, replace=True)
        X_base = X_tgt[idx]
        X_p = (1 - self.alpha) * X_base + self.alpha * centroid_src
        # Небольшой шум, чтобы объекты не были идентичны
        X_p += rng.normal(0, 0.01 * (X.std(axis=0) + 1e-9), size=X_p.shape)
        y_p = np.full(len(X_p), self.target_class, dtype=y.dtype)
        return X_p, y_p


# ---------------------------------------------------------------------------
# Gradient-based (back-gradient) атака
# ---------------------------------------------------------------------------

class OptimizationBasedAttack(BaseAttack):
    """Приближённая градиентная атака на логистическую регрессию.

    Реализует упрощённую версию back-gradient poisoning (Biggio,
    Muñoz-González и др.): вместо точного решения bilevel-задачи
    используется эвристика на основе градиента модели-суррогата.

    Идея: обучить вспомогательную логистическую регрессию на чистых
    данных, а затем сместить отравлённые точки в направлении,
    максимизирующем потери на валидационной выборке.

    Замечание: полноценная реализация back-gradient требует
    дифференцируемой процедуры оптимизации (на практике часто
    используют SGD с малым числом эпох); здесь реализация
    ограничена первым порядком.
    """

    name = "optimization_based"

    def __init__(self, n_steps: int = 20, lr: float = 0.1,
                 target_class: int = 1):
        self.n_steps = n_steps
        self.lr = lr
        self.target_class = target_class

    def generate(self, X, y, n_poison, seed=42):
        rng = np.random.default_rng(seed)
        # Суррогат: логистическая регрессия на чистых данных
        lr_model = LogisticRegression(max_iter=1000).fit(X, y)
        w = lr_model.coef_.ravel()
        b = lr_model.intercept_[0]

        # Стартовые точки: выбираем объекты НЕ target_class
        src_mask = y != self.target_class
        idx = rng.choice(np.where(src_mask)[0], size=n_poison, replace=True)
        X_p = X[idx].copy()
        # Ложная метка — target_class
        y_p = np.full(n_poison, self.target_class, dtype=y.dtype)

        # Градиентное смещение. Для логистической регрессии
        # логит l = w^T x + b, вероятность σ(l).
        # Потеря для точки (x, y') = -log σ((2y'-1) * l).
        # d loss / d x = -(2y'-1) * (1 - σ((2y'-1)l)) * w.
        # Атакующий МАКСИМИЗИРУЕТ расхождение модели на валидации,
        # поэтому двигает точки так, чтобы их метка становилась
        # "крепче" (модель теснее притягивается к ним) —
        # т. е. В сторону уменьшения loss на ОТРАВЛЕННЫХ точках.
        # Это приближение: точная bilevel формула сложнее.
        y_signed = (2 * y_p - 1)  # для бинарной задачи 0/1 → -1/+1
        for _ in range(self.n_steps):
            logit = X_p @ w + b
            prob = 1.0 / (1.0 + np.exp(-y_signed * logit))
            grad = -(y_signed * (1 - prob))[:, None] * w[None, :]
            X_p -= self.lr * grad  # шаг против градиента loss
        return X_p, y_p


# ---------------------------------------------------------------------------
# Backdoor / trigger-based
# ---------------------------------------------------------------------------

class BackdoorAttack(BaseAttack):
    """Бэкдор-атака с фиксированным триггером в признаках.

    Атакующий подмешивает в обучающую выборку объекты произвольных
    классов, к которым добавлен фиксированный триггер
    (значения в `trigger_indices` установлены в `trigger_value`),
    а метка заменена на `target_class`. При тестировании любая
    точка с тем же триггером будет с высокой вероятностью
    классифицирована как `target_class`.
    """

    name = "backdoor"

    def __init__(self, trigger_indices: Optional[list] = None,
                 trigger_value: float = 3.0,
                 target_class: int = 1):
        self.trigger_indices = trigger_indices
        self.trigger_value = trigger_value
        self.target_class = target_class

    def generate(self, X, y, n_poison, seed=42):
        rng = np.random.default_rng(seed)
        n_features = X.shape[1]
        trig = self.trigger_indices or list(range(min(3, n_features)))
        idx = rng.choice(len(X), size=n_poison, replace=True)
        X_p = X[idx].copy()
        X_p[:, trig] = self.trigger_value
        y_p = np.full(n_poison, self.target_class, dtype=y.dtype)
        return X_p, y_p

    def apply_trigger(self, X: np.ndarray) -> np.ndarray:
        """Применяет триггер к произвольной выборке (для оценки ASR)."""
        n_features = X.shape[1]
        trig = self.trigger_indices or list(range(min(3, n_features)))
        X_t = X.copy()
        X_t[:, trig] = self.trigger_value
        return X_t
