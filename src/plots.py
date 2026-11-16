"""Figuras dos experimentos, sempre calculadas das saídas reais."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GRAY = "#172d43", "#087f8c", "#697b8b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": NAVY, "text.color": NAVY,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def make_figures(frame, report, histories, destination):
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    counts = frame.alvo.value_counts().reindex([0, 1])
    bars = axes[0].bar(["Furto", "Roubo"], counts, color=[TEAL, NAVY], width=.55)
    axes[0].bar_label(bars, padding=4, fmt="%d")
    axes[0].set_ylim(0, counts.max()*1.18)
    axes[0].set_title("Classes na amostra de 20.000 BOs")
    axes[0].set_ylabel("Boletins únicos")
    frame.regiao.value_counts().sort_values().plot.barh(ax=axes[1], color=TEAL)
    axes[1].set_xlabel("Boletins únicos")
    axes[1].set_ylabel("")
    axes[1].set_title("Região da ocorrência")
    fig.savefig(dest / "exploracao.png", dpi=180)
    plt.close(fig)

    selected = next(c for c in report["configs"] if c["name"] == report["selected"])
    history = pd.DataFrame(histories[report["selected"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, metric, title, ylabel in zip(axes, ["accuracy", "loss"],
                                        ["Acurácia por época", "Perda por época"],
                                        ["Acurácia", "Entropia cruzada binária"]):
        ax.plot(history.epoch, history[f"train_{metric}"], color=NAVY, label="Treino", lw=2)
        ax.plot(history.epoch, history[f"val_{metric}"], color=TEAL, label="Validação", lw=2)
        ax.axvline(selected["best_epoch"], color=GRAY, linestyle="--", lw=1,
                   label=f"Checkpoint: época {selected['best_epoch']}")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=.12)
    fig.savefig(dest / "curvas_modelo_selecionado.png", dpi=200)
    plt.close(fig)

    matrix = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1)*1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=24,
                    color="white" if matrix[i,j]>matrix.max()*.65 else NAVY, weight="bold")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Furto", "Roubo"],
           yticklabels=["Furto", "Roubo"], xlabel="Classe estimada", ylabel="Classe real",
           title="Matriz de confusão · teste (n = 3.000)")
    ax.tick_params(length=0)
    fig.savefig(dest / "matriz_confusao.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.3), constrained_layout=True)
    for result in report["configs"]:
        data = pd.DataFrame(histories[result["name"]])
        ax.plot(data.epoch, data.val_loss, label=result["name"], lw=2)
    ax.set(xlabel="Época", ylabel="Entropia cruzada", title="Perda de validação das três arquiteturas")
    ax.legend(frameon=False)
    ax.grid(alpha=.12)
    fig.savefig(dest / "comparacao_curvas.png", dpi=180)
    plt.close(fig)
