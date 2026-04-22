"""
Полный сравнительный эксперимент: 3 датасета × 4 атаки × 5 защит
× 3 уровня отравления. Результаты сохраняются в CSV и JSON.

Запуск: python experiments/run_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poisondefense import (  # noqa: E402
    AnomalyDetectionDefense,
    BackdoorAttack,
    FeatureCollisionAttack,
    HybridPoisoningDefense,
    LabelFlipAttack,
    OptimizationBasedAttack,
    RONIDefense,
    TargetedLabelFlipAttack,
    TrimmedLossDefense,
    attack_success_rate,
    clean_accuracy,
    detection_metrics,
    inject_poison,
    load_dataset,
    set_random_state,
    split_train_val_test,
)
from sklearn.base import clone


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_defenses(base, contamination: float):
    return {
        "none": None,
        "anomaly_iforest": AnomalyDetectionDefense(
            base_estimator=base, method="isoforest",
            contamination=contamination),
        "anomaly_lof": AnomalyDetectionDefense(
            base_estimator=base, method="lof",
            contamination=contamination),
        "roni": RONIDefense(base_estimator=base, batch_size=10),
        "trimmed_loss": TrimmedLossDefense(
            base_estimator=base, trim_ratio=contamination, n_iter=5),
        "hybrid": HybridPoisoningDefense(
            base_estimator=base, contamination=contamination,
            use_anomaly=True, use_roni=True, use_trimmed_loss=True),
    }


def make_attacks():
    return {
        "label_flip": LabelFlipAttack(),
        "targeted_flip": TargetedLabelFlipAttack(
            source_class=0, target_class=1, strategy="nearest"),
        "feature_collision": FeatureCollisionAttack(
            source_class=0, target_class=1, alpha=0.6),
        "backdoor": BackdoorAttack(trigger_value=3.0, target_class=1),
    }


def evaluate(defense_name, defense, X_tr, y_tr, is_poison,
             X_te, y_te, base):
    """Возвращает dict: clean_acc, detection-метрики, keep-доля."""
    t0 = time.time()
    if defense is None:
        model = clone(base).fit(X_tr, y_tr)
        keep = np.ones(len(X_tr), dtype=bool)
    else:
        defense.fit(X_tr, y_tr)
        model = defense.model_
        keep = defense.keep_mask_
    t_fit = time.time() - t0
    acc = clean_accuracy(model, X_te, y_te)
    det = detection_metrics(is_poison, ~keep)
    return {
        "defense": defense_name,
        "clean_acc": acc,
        "fit_time_sec": t_fit,
        "keep_fraction": float(keep.mean()),
        **{f"det_{k}": v for k, v in det.items()},
        "model": model,
        "keep": keep,
    }


def run_one(ds_name, attack_name, poison_rate, seed=42):
    set_random_state(seed)
    ds = load_dataset(ds_name, binary=True)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_train_val_test(
        ds.X, ds.y, seed=seed)

    attacks = make_attacks()
    atk = attacks[attack_name]
    X_mix, y_mix, is_poison = inject_poison(
        X_tr, y_tr, atk, poison_rate=poison_rate, seed=seed)

    base = LogisticRegression(max_iter=2000)
    defenses = make_defenses(base, contamination=max(0.02, poison_rate))

    # baseline без атаки
    clean_model = clone(base).fit(X_tr, y_tr)
    acc_clean_no_attack = clean_accuracy(clean_model, X_te, y_te)

    results = []
    # ASR: для targeted и backdoor — вычисляем на тест.объектах НЕ target_class
    def asr_for(model):
        if attack_name == "backdoor":
            X_trig = atk.apply_trigger(X_te[y_te != 1])
            return attack_success_rate(model, X_trig, y_target=1)
        if attack_name in ("targeted_flip", "feature_collision"):
            X_src = X_te[y_te == 0]
            return attack_success_rate(model, X_src, y_target=1)
        # availability: ASR не определён — возвращаем 1 - accuracy
        return 1.0 - clean_accuracy(model, X_te, y_te)

    for dname, dobj in defenses.items():
        res = evaluate(dname, dobj, X_mix, y_mix, is_poison,
                       X_te, y_te, base)
        res["asr"] = asr_for(res["model"])
        res["dataset"] = ds_name
        res["attack"] = attack_name
        res["poison_rate"] = poison_rate
        res["acc_clean_no_attack"] = acc_clean_no_attack
        # убираем тяжёлые объекты
        res.pop("model", None)
        res.pop("keep", None)
        results.append(res)
    return results


def main():
    datasets = ["breast_cancer", "spambase", "synthetic"]
    attacks = ["label_flip", "targeted_flip", "feature_collision", "backdoor"]
    poison_rates = [0.05, 0.10, 0.20]
    seeds = [42, 123, 2024]

    all_rows = []
    total = len(datasets) * len(attacks) * len(poison_rates) * len(seeds)
    done = 0
    for ds, atk, pr, s in product(datasets, attacks, poison_rates, seeds):
        try:
            rows = run_one(ds, atk, pr, seed=s)
            for r in rows:
                r["seed"] = s
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] {ds}/{atk}/{pr}/seed={s}: {e}")
        done += 1
        if done % 5 == 0:
            print(f"  progress: {done}/{total}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "experiments_raw.csv", index=False)

    # Агрегация: среднее по seed
    agg = (df.groupby(["dataset", "attack", "poison_rate", "defense"])
             .agg(clean_acc=("clean_acc", "mean"),
                  clean_acc_std=("clean_acc", "std"),
                  asr=("asr", "mean"),
                  det_precision=("det_precision", "mean"),
                  det_recall=("det_recall", "mean"),
                  det_f1=("det_f1", "mean"),
                  fit_time_sec=("fit_time_sec", "mean"))
             .reset_index())
    agg.to_csv(RESULTS_DIR / "experiments_agg.csv", index=False)

    # Глобальный ранжирующий скоринг защит:
    # мы ищем защиту с высоким clean_acc и низким ASR в среднем.
    rank_df = (df.groupby("defense")
                 .agg(clean_acc=("clean_acc", "mean"),
                      asr=("asr", "mean"),
                      det_f1=("det_f1", "mean"),
                      fit_time_sec=("fit_time_sec", "mean"))
                 .reset_index())
    rank_df["score"] = rank_df["clean_acc"] - rank_df["asr"]
    rank_df = rank_df.sort_values("score", ascending=False)
    rank_df.to_csv(RESULTS_DIR / "defense_ranking.csv", index=False)

    summary = {
        "n_runs": len(df),
        "datasets": datasets,
        "attacks": attacks,
        "poison_rates": poison_rates,
        "seeds": seeds,
        "best_defense_by_score": rank_df.iloc[0]["defense"],
        "ranking": rank_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Итоговый рейтинг защит ===")
    print(rank_df.to_string(index=False))
    print(f"\nЛучшая защита по (clean_acc - ASR): {rank_df.iloc[0]['defense']}")


if __name__ == "__main__":
    main()
