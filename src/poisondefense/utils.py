"""
Вспомогательные утилиты: загрузка датасетов, разбиение выборок,
внедрение отравления в обучающие данные, фиксация генератора
случайных чисел для воспроизводимости.

Все функции снабжены подробными русскоязычными комментариями,
чтобы код был понятен читателю без глубокой ML-подготовки.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------

def set_random_state(seed: int = 42) -> None:
    """Фиксирует ГПСЧ для воспроизводимости экспериментов.

    Parameters
    ----------
    seed : int
        Значение seed, которое будет установлено для numpy, random
        и переменной окружения PYTHONHASHSEED. По умолчанию 42.
    """
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Загрузка датасетов
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Контейнер для датасета с метаданными."""
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    target_names: list
    name: str

    def __repr__(self) -> str:
        return (f"Dataset(name={self.name!r}, n={len(self.X)}, "
                f"d={self.X.shape[1]}, classes={len(set(self.y))})")


def load_dataset(name: str, binary: bool = True,
                 n_samples: int = 2000, seed: int = 42) -> Dataset:
    """Загружает стандартный датасет из scikit-learn / OpenML.

    Parameters
    ----------
    name : str
        Идентификатор датасета. Допустимые значения:

        - "breast_cancer" — Wisconsin Breast Cancer (бинарная классификация);
        - "iris"          — Iris (3 класса; при binary=True оставляем 2);
        - "wine"          — Wine recognition;
        - "digits"        — рукописные цифры 8×8;
        - "spambase"      — Spambase из OpenML (детектор спама);
        - "synthetic"     — синтетические данные make_classification.
    binary : bool
        Если True, для многоклассовых датасетов оставляем только
        два первых класса (иначе многие атаки становятся
        плохо определёнными).
    n_samples : int
        Используется только для "synthetic".
    seed : int
        Seed для генератора случайных чисел.

    Returns
    -------
    Dataset
        Контейнер с матрицей признаков X, вектором меток y,
        именами признаков и классов.
    """
    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data.astype(np.float64), d.target.astype(int)
        features = list(d.feature_names)
        targets = list(d.target_names)
    elif name == "iris":
        d = load_iris()
        X, y = d.data.astype(np.float64), d.target.astype(int)
        features = list(d.feature_names)
        targets = list(d.target_names)
    elif name == "wine":
        d = load_wine()
        X, y = d.data.astype(np.float64), d.target.astype(int)
        features = list(d.feature_names)
        targets = list(d.target_names)
    elif name == "digits":
        d = load_digits()
        X, y = d.data.astype(np.float64), d.target.astype(int)
        features = [f"pix{i}" for i in range(X.shape[1])]
        targets = [str(i) for i in range(10)]
    elif name == "spambase":
        # Spambase из UCI через OpenML; кешируется в ~/scikit_learn_data
        d = fetch_openml("spambase", version=1, as_frame=False,
                         parser="liac-arff")
        X = d.data.astype(np.float64)
        y = d.target.astype(int)
        features = list(d.feature_names) if d.feature_names else \
            [f"f{i}" for i in range(X.shape[1])]
        targets = ["ham", "spam"]
    elif name == "synthetic":
        X, y = make_classification(
            n_samples=n_samples, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, class_sep=1.0,
            flip_y=0.01, random_state=seed,
        )
        X = X.astype(np.float64)
        features = [f"x{i}" for i in range(X.shape[1])]
        targets = ["neg", "pos"]
    else:
        raise ValueError(f"Неизвестный датасет: {name!r}")

    if binary and len(set(y)) > 2:
        mask = y < 2
        X, y = X[mask], y[mask]
        targets = targets[:2]

    return Dataset(X=X, y=y, feature_names=features,
                   target_names=targets, name=name)


# ---------------------------------------------------------------------------
# Разбиение
# ---------------------------------------------------------------------------

def split_train_val_test(
    X: np.ndarray, y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    standardize: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Делит выборку на train / val / test со стратификацией.

    Возвращает (X_tr, y_tr, X_val, y_val, X_te, y_te). При
    standardize=True применяется StandardScaler, обученный только
    на трейне (fit_on_train), чтобы не было утечки данных.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8, \
        "Сумма долей должна быть равна 1.0"

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)

    rel_val = val_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=rel_val,
        stratify=y_rest, random_state=seed)

    if standardize:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_val, X_te = sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te)

    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ---------------------------------------------------------------------------
# Внедрение отравления
# ---------------------------------------------------------------------------

def inject_poison(
    X_clean: np.ndarray, y_clean: np.ndarray,
    attack: "BaseAttack",
    poison_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Внедряет отравление в обучающую выборку.

    Parameters
    ----------
    X_clean, y_clean : np.ndarray
        Чистая обучающая выборка.
    attack : BaseAttack
        Экземпляр атаки (см. модуль `attacks`).
    poison_rate : float
        Доля отравлённых объектов от размера чистой выборки.
    seed : int
        Seed для воспроизводимости.

    Returns
    -------
    X_mix, y_mix : np.ndarray
        Смешанная выборка (чистые + отравлённые объекты).
    is_poison : np.ndarray of bool
        Булев массив длины len(X_mix): True, если объект отравлён.
    """
    n_poison = int(round(poison_rate * len(X_clean)))
    X_p, y_p = attack.generate(X_clean, y_clean, n_poison=n_poison, seed=seed)
    X_mix = np.vstack([X_clean, X_p])
    y_mix = np.concatenate([y_clean, y_p])
    is_poison = np.concatenate([
        np.zeros(len(X_clean), dtype=bool),
        np.ones(len(X_p), dtype=bool),
    ])
    # Перемешаем, чтобы отравлённые объекты не шли блоком в конце
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm], is_poison[perm]
