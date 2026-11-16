"""Gera metadados e copia a aplicação estática para dist/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import shutil
import pandas as pd
from src.data import ROOT, split_data


def build_web():
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    frame = pd.read_csv(ROOT / "data/processed/amostra_capital_2016.csv")
    train, _, _ = split_data(frame)
    selected = next(c for c in report["configs"] if c["name"] == report["selected"])
    gap = selected["training"]["accuracy"] - selected["validation"]["accuracy"]
    note = (f"No checkpoint, a acurácia de treino é {selected['training']['accuracy']:.1%} e a de validação é {selected['validation']['accuracy']:.1%}, "
            f"uma diferença de {gap*100:.2f} pontos percentuais. O gap reduzido não sugere overfitting forte. "
            "O desempenho moderado nas duas partes pode refletir informação limitada nas duas entradas. "
            "A parada antecipada restaura o melhor checkpoint de validação. Uma semente e uma divisão não permitem conclusões de significância.")
    metadata = {"report": report, "overfitting_note": note,
                "training_support": train.groupby(["regiao", "contexto"]).size().reset_index(name="count").to_dict("records")}
    (ROOT / "web/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    shutil.copytree(ROOT / "reports/figures", ROOT / "web/figures", dirs_exist_ok=True)
    shutil.copytree(ROOT / "web", ROOT / "dist", dirs_exist_ok=True)


if __name__ == "__main__":
    build_web()
