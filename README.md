# Методика защиты классических моделей машинного обучения от атак отравления данных

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Репозиторий сопровождает магистерскую ВКР на тему:

> **«Методика защиты моделей машинного обучения от атак с использованием отравления данных»**

Содержит:

- `src/poisondefense/` — программный модуль на Python с реализацией 5 атак и 4 защит (в т. ч. авторской гибридной);
- `notebooks/experiments.ipynb` — подробный экспериментальный Jupyter-ноутбук (33 ячейки, понятный «с нуля»);
- `experiments/` — скрипты массовых экспериментов;
- `results/` — сырые и агрегированные результаты, графики;
- `tests/` — набор unit-тестов;
- `docs/` — руководство по эксплуатации.

## Содержание

1. [Установка](#установка)
2. [Быстрый старт](#быстрый-старт)
3. [Архитектура модуля](#архитектура-модуля)
4. [Поддерживаемые атаки](#поддерживаемые-атаки)
5. [Поддерживаемые защиты](#поддерживаемые-защиты)
6. [Воспроизведение экспериментов](#воспроизведение-экспериментов)
7. [Результаты](#результаты)
8. [Цитирование](#цитирование)

## Установка

```bash
git clone https://github.com/Dnlln/thesis-ml-data-poisoning-defense.git
cd thesis-ml-data-poisoning-defense
pip install -e .
# или без установки пакета:
pip install -r requirements.txt
export PYTHONPATH=src
```

**Требования:** Python ≥ 3.10, `numpy`, `pandas`, `scikit-learn ≥ 1.3`, `matplotlib`, `seaborn`.

## Быстрый старт

```python
from sklearn.linear_model import LogisticRegression
from poisondefense import (
    load_dataset, split_train_val_test, inject_poison,
    LabelFlipAttack, HybridPoisoningDefense, clean_accuracy,
)

# 1) Данные
ds = load_dataset("breast_cancer")
X_tr, y_tr, X_val, y_val, X_te, y_te = split_train_val_test(ds.X, ds.y, seed=42)

# 2) Атака: переворачиваем 15% меток
attack = LabelFlipAttack()
X_mix, y_mix, is_poison = inject_poison(X_tr, y_tr, attack, poison_rate=0.15)

# 3) Авторская гибридная защита
defense = HybridPoisoningDefense(
    base_estimator=LogisticRegression(max_iter=2000),
    contamination=0.15)
defense.fit(X_mix, y_mix)

print("Точность с защитой:", clean_accuracy(defense, X_te, y_te))
```

## Архитектура модуля

```
src/poisondefense/
├── __init__.py        публичное API
├── attacks.py         5 атак отравления
├── defenses.py        4 защиты + HybridPoisoningDefense (авторская)
├── metrics.py         clean_accuracy, attack_success_rate, detection_metrics
└── utils.py           load_dataset, split_train_val_test, inject_poison
```

## Поддерживаемые атаки

| Класс                      | Тип                     | Примечание                                      |
|---------------------------|-------------------------|-------------------------------------------------|
| `LabelFlipAttack`         | availability (dirty)    | Случайная смена меток                           |
| `TargetedLabelFlipAttack` | targeted (dirty)        | Ближайшие к границе точки + смена меток         |
| `FeatureCollisionAttack`  | targeted (clean-label)  | Сдвиг объектов target-класса в сторону source   |
| `OptimizationBasedAttack` | targeted (dirty)        | Упрощённая back-gradient атака на LogReg        |
| `BackdoorAttack`          | targeted (dirty, trig.) | Закладывает триггер в обучение                  |

## Поддерживаемые защиты

| Класс                        | Идея                                             |
|------------------------------|--------------------------------------------------|
| `AnomalyDetectionDefense`    | IsolationForest / LOF / EllipticEnvelope по классам |
| `RONIDefense`                | Reject On Negative Impact (батчевая)              |
| `TrimmedLossDefense`         | Итеративная обрезка top-α потерь                  |
| **`HybridPoisoningDefense`** | **Авторская методика:** AD → RONI → Trimmed-Loss  |

Все защиты имеют `fit/predict` интерфейс, совместимый с scikit-learn, и метод `sanitize(X, y)` для получения маски «чистых» объектов.

## Воспроизведение экспериментов

```bash
# Полный прогон (3 датасета × 4 атаки × 3 доли × 3 seed = 108 конфигураций)
python experiments/run_experiments.py

# Графики
python experiments/make_plots.py

# Тесты
pytest tests/ -v
```

## Результаты

Сводный рейтинг защит (среднее по 108 прогонам, метрика `score = clean_acc − ASR`):

| Защита               | Точность | ASR    | F1 дет. | Score  |
|----------------------|----------|--------|---------|--------|
| **Гибридная (авт.)** | 0.891    | 0.258  | 0.208   | **0.633** |
| Isolation Forest     | 0.894    | 0.263  | 0.201   | 0.631   |
| Trimmed-Loss         | 0.894    | 0.281  | 0.401   | 0.614   |
| LOF                  | 0.894    | 0.294  | 0.144   | 0.601   |
| Без защиты           | 0.891    | 0.324  | —       | 0.567   |
| RONI                 | 0.885    | 0.326  | 0.157   | 0.559   |

![Сравнение защит](results/figures/defense_ranking.png)
![Устойчивость по доле отравления](results/figures/poison_rate_dependency.png)

**Вывод:** гибридная методика даёт лучший усреднённый баланс «точность — устойчивость» за счёт декомпозиции очистки на три комплементарных этапа.

## Цитирование

Если вы используете материалы репозитория в своей работе:

```bibtex
@mastersthesis{thesis2026mldatapoisoning,
  author = {{Ильин, Д.}},
  title  = {Методика защиты моделей машинного обучения от атак с использованием отравления данных},
  school = {Магистерская ВКР},
  year   = {2026}
}
```

## Лицензия

MIT (см. `LICENSE`).
