# Руководство по эксплуатации модуля `poisondefense`

Версия: 1.0.0 · Автор: ВКР 2026

Настоящее руководство предназначено для пользователей программного модуля `poisondefense`, реализующего авторскую методику защиты классических моделей машинного обучения от атак отравления данных.

## 1. Назначение

Модуль предоставляет:

1. Библиотеку **атак отравления данных** (5 типов), воспроизводящих основные класс-атаки из литературы: label flipping (random/targeted), feature collision (clean-label), gradient-based и backdoor;
2. Библиотеку **защитных механизмов** (4 метода), включая авторскую гибридную методику;
3. Вспомогательные утилиты для загрузки стандартных датасетов, стратифицированного разбиения и воспроизводимого внедрения отравления;
4. Метрики качества (clean accuracy, ASR, метрики детекции).

## 2. Системные требования

| Компонент     | Минимально         |
|---------------|--------------------|
| Python        | 3.10               |
| numpy         | 1.26               |
| scikit-learn  | 1.3                |
| pandas        | 2.0                |
| matplotlib    | 3.7 (для графиков) |
| RAM           | 2 ГБ               |
| Диск          | 50 МБ              |

## 3. Установка

### Вариант A — через pip (рекомендуется)

```bash
git clone https://github.com/Dnlln/thesis-ml-data-poisoning-defense.git
cd thesis-ml-data-poisoning-defense
pip install -e .
```

### Вариант B — без установки пакета

```bash
pip install -r requirements.txt
export PYTHONPATH=src         # Linux / macOS
set PYTHONPATH=src            # Windows cmd
$env:PYTHONPATH = "src"       # Windows PowerShell
```

## 4. Быстрый старт

```python
from sklearn.linear_model import LogisticRegression
from poisondefense import (
    load_dataset, split_train_val_test, inject_poison,
    LabelFlipAttack, HybridPoisoningDefense, clean_accuracy,
)

# 1. Данные (встроенные датасеты: breast_cancer, iris, wine,
#    digits, spambase, synthetic)
ds = load_dataset("breast_cancer")
X_tr, y_tr, X_val, y_val, X_te, y_te = split_train_val_test(ds.X, ds.y)

# 2. Внедрение отравления
atk = LabelFlipAttack()
X_mix, y_mix, is_poison = inject_poison(X_tr, y_tr, atk, poison_rate=0.15)

# 3. Защита
defense = HybridPoisoningDefense(
    base_estimator=LogisticRegression(max_iter=2000),
    contamination=0.15)
defense.fit(X_mix, y_mix)

# 4. Метрика
print("Точность с защитой:", clean_accuracy(defense, X_te, y_te))
```

## 5. Справочник API

### 5.1. Утилиты

#### `load_dataset(name, binary=True, n_samples=2000, seed=42) → Dataset`
Загружает датасет. Допустимые `name`: `"breast_cancer"`, `"iris"`, `"wine"`, `"digits"`, `"spambase"`, `"synthetic"`.
Возвращает объект `Dataset(X, y, feature_names, target_names, name)`.

#### `split_train_val_test(X, y, train_size=0.6, val_size=0.2, test_size=0.2, standardize=True, seed=42)`
Стратифицированное разбиение на train/val/test; при `standardize=True` — стандартизация, обученная только на train.

#### `inject_poison(X_clean, y_clean, attack, poison_rate=0.1, seed=42) → (X_mix, y_mix, is_poison)`
Добавляет `int(round(poison_rate * n))` отравлённых объектов, сгенерированных `attack`, к чистой выборке; возвращает перемешанную смесь и булев массив истинных отравлений.

#### `set_random_state(seed=42)`
Фиксирует seed для numpy / random / PYTHONHASHSEED.

### 5.2. Атаки

Все атаки наследуются от `BaseAttack` и имеют метод `generate(X, y, n_poison, seed=42) → (X_p, y_p)`.

| Класс | Ключевые параметры | Модель угроз |
|-------|--------------------|--------------|
| `LabelFlipAttack()` | — | availability, dirty-label |
| `TargetedLabelFlipAttack(source_class, target_class, strategy)` | `strategy ∈ {"nearest","random"}` | targeted, dirty-label |
| `FeatureCollisionAttack(source_class, target_class, alpha)` | `alpha ∈ [0,1]` — степень сдвига к чужому центроиду | targeted, clean-label |
| `OptimizationBasedAttack(n_steps, lr, target_class)` | шаги/LR градиента | targeted, dirty-label |
| `BackdoorAttack(trigger_indices, trigger_value, target_class)` | индексы и значение триггера | backdoor |

У `BackdoorAttack` есть дополнительный метод `apply_trigger(X)` для применения триггера к любой выборке — используется для оценки ASR.

### 5.3. Защиты

Все защиты совместимы с sklearn: имеют `fit(X, y)`, `predict(X)`, `predict_proba(X)`, дополнительный `sanitize(X, y) → keep_mask` и атрибут `keep_mask_` после обучения.

#### `AnomalyDetectionDefense(base_estimator, method, contamination, per_class, random_state)`
- `method ∈ {"isoforest", "lof", "elliptic"}`
- `per_class=True` (по умолчанию) — детектор обучается независимо для каждого класса.

#### `RONIDefense(base_estimator, val_fraction, batch_size, threshold, random_state)`
Батчевая RONI. При `batch_size=1` — оригинальная объектная версия (медленнее).

#### `TrimmedLossDefense(base_estimator, trim_ratio, n_iter)`
Итеративная обрезка top-α объектов по log-loss; `trim_ratio` соответствует ожидаемой доле отравления.

#### `HybridPoisoningDefense(base_estimator, contamination, use_anomaly, use_roni, use_trimmed_loss, anomaly_method, roni_batch_size, trim_ratio, trim_n_iter, random_state)`
Авторская трёхступенчатая схема. Флаги `use_*` позволяют отключать отдельные этапы для ablation-экспериментов.

### 5.4. Метрики

- `clean_accuracy(model, X_test, y_test)` — точность на чистом тесте.
- `attack_success_rate(model, X_attack, y_target)` — доля объектов, предсказанных как `y_target`.
- `detection_metrics(is_poison_true, is_poison_pred)` — precision / recall / F1 / ROC-AUC детекции.

## 6. Примеры типовых сценариев

### 6.1. Сравнение нескольких защит

```python
from poisondefense import (AnomalyDetectionDefense, RONIDefense,
                           TrimmedLossDefense, HybridPoisoningDefense)

defenses = {
    "AD": AnomalyDetectionDefense(method="isoforest"),
    "RONI": RONIDefense(),
    "TRIM": TrimmedLossDefense(trim_ratio=0.15),
    "Hybrid": HybridPoisoningDefense(contamination=0.15),
}
for name, d in defenses.items():
    d.fit(X_mix, y_mix)
    print(name, clean_accuracy(d, X_te, y_te))
```

### 6.2. Оценка устойчивости к backdoor

```python
atk = BackdoorAttack(trigger_indices=[0,1,2], trigger_value=3.0, target_class=1)
X_mix, y_mix, _ = inject_poison(X_tr, y_tr, atk, poison_rate=0.1)

defense = HybridPoisoningDefense(contamination=0.1).fit(X_mix, y_mix)

X_trig = atk.apply_trigger(X_te[y_te != 1])
asr = attack_success_rate(defense, X_trig, y_target=1)
print(f"ASR после защиты: {asr:.3f}")
```

### 6.3. Массовый эксперимент

Готовый скрипт `experiments/run_experiments.py` прогоняет 3 датасета × 4 атаки × 3 уровня отравления × 3 seed = 108 конфигураций и сохраняет CSV/JSON в `results/`.

## 7. Производительность

Время обучения на одном конфиге (breast_cancer, LogReg, 341 train):

| Защита          | Время (сек) |
|-----------------|-------------|
| Без защиты      | 0.005       |
| Isolation Forest| 0.22        |
| LOF             | 0.03        |
| RONI            | 0.34        |
| Trimmed-Loss    | 0.03        |
| **Hybrid**      | **0.51**    |

Гибридная защита дороже одиночных в 2–10 раз, но это оправдано выигрышем качества (см. раздел «Результаты» в README).

## 8. Известные ограничения

1. `OptimizationBasedAttack` реализует **упрощённую** версию back-gradient без полной дифференциации внутреннего цикла.
2. `RONIDefense` имеет сложность O(N / batch_size) обучений базовой модели — для больших данных используйте бóльший `batch_size`.
3. Детекторы аномалий работают на сырых признаках. При очень высокой размерности рекомендуется предварительное понижение размерности (PCA).

## 9. Поддержка

Вопросы и баг-репорты — через Issues репозитория <https://github.com/Dnlln/thesis-ml-data-poisoning-defense>.

## 10. Лицензия

Модуль распространяется под лицензией MIT (см. `LICENSE`).
