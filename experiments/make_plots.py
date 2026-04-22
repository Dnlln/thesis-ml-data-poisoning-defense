"""
Построение графиков по результатам экспериментов.
Все рисунки сохраняются в results/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

DEFENSE_ORDER = ["none", "anomaly_iforest", "anomaly_lof", "roni",
                 "trimmed_loss", "hybrid"]
DEFENSE_LABELS = {
    "none": "Без защиты",
    "anomaly_iforest": "Isolation Forest",
    "anomaly_lof": "LOF",
    "roni": "RONI",
    "trimmed_loss": "Trimmed-Loss",
    "hybrid": "Гибридная (авт.)",
}
ATTACK_LABELS = {
    "label_flip": "Случайная перестановка меток",
    "targeted_flip": "Целевая перестановка меток",
    "feature_collision": "Clean-label коллизия",
    "backdoor": "Бэкдор",
}

df = pd.read_csv(RES / "experiments_agg.csv")

# 1. Barplot: clean_acc vs defense, по атакам (poison=0.10)
def plot_clean_acc_by_attack():
    sub = df[df.poison_rate == 0.10].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = sub.pivot_table(index="attack", columns="defense",
                            values="clean_acc")
    pivot = pivot[DEFENSE_ORDER]
    pivot.index = [ATTACK_LABELS[a] for a in pivot.index]
    pivot.columns = [DEFENSE_LABELS[d] for d in pivot.columns]
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Точность на чистом тесте")
    ax.set_xlabel("")
    ax.set_title("Точность классификации при разных атаках и защитах (poison = 10%)")
    ax.set_ylim(0.7, 1.0)
    ax.legend(title="Защита", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "clean_acc_by_attack.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_asr_by_attack():
    sub = df[df.poison_rate == 0.10].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = sub.pivot_table(index="attack", columns="defense", values="asr")
    pivot = pivot[DEFENSE_ORDER]
    pivot.index = [ATTACK_LABELS[a] for a in pivot.index]
    pivot.columns = [DEFENSE_LABELS[d] for d in pivot.columns]
    pivot.plot(kind="bar", ax=ax, width=0.8, colormap="Reds_r")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_xlabel("")
    ax.set_title("Успешность атаки при разных защитах (poison = 10%)")
    ax.legend(title="Защита", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "asr_by_attack.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_poison_rate_dependency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
        axes, ["clean_acc", "asr"],
        ["Точность (выше — лучше)", "ASR (ниже — лучше)"],
    ):
        g = df.groupby(["poison_rate", "defense"])[metric].mean().reset_index()
        for d in DEFENSE_ORDER:
            sd = g[g.defense == d]
            ax.plot(sd.poison_rate, sd[metric], marker="o",
                    label=DEFENSE_LABELS[d])
        ax.set_xlabel("Доля отравленных объектов")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "poison_rate_dependency.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_detection_f1():
    sub = df[df.poison_rate == 0.10].copy()
    pivot = sub.pivot_table(index="attack", columns="defense", values="det_f1")
    # none — не делает детекции, исключим
    cols = [c for c in DEFENSE_ORDER if c != "none"]
    pivot = pivot[cols]
    pivot.index = [ATTACK_LABELS[a] for a in pivot.index]
    pivot.columns = [DEFENSE_LABELS[d] for d in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "F1 детекции"})
    ax.set_title("F1 детекции отравлённых объектов (poison = 10%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "detection_f1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranking():
    r = pd.read_csv(RES / "defense_ranking.csv")
    r["defense_ru"] = r["defense"].map(DEFENSE_LABELS)
    r = r.sort_values("score")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#d62728" if d == "Гибридная (авт.)" else "#1f77b4"
              for d in r.defense_ru]
    ax.barh(r.defense_ru, r.score, color=colors)
    ax.set_xlabel("Сводный индекс (clean_acc − ASR)")
    ax.set_title("Сводный рейтинг защит (усреднение по всем экспериментам)")
    for i, (n, s) in enumerate(zip(r.defense_ru, r.score)):
        ax.text(s + 0.005, i, f"{s:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(FIG / "defense_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_clean_acc_by_attack()
    plot_asr_by_attack()
    plot_poison_rate_dependency()
    plot_detection_f1()
    plot_ranking()
    print("Графики сохранены в", FIG)
    for f in sorted(FIG.iterdir()):
        print(" ", f.name)
