"""Executar da raiz: python scripts/train.py. Reproduz todos os artefatos da MLP."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from src.data import ROOT, download_raw, prepare_data
from src.model import run_experiments, save_outputs
from src.plots import make_figures

if __name__ == "__main__":
    frame, audit = prepare_data(download_raw())
    frame.to_csv(ROOT / "data/processed/amostra_capital_2016.csv", index=False)
    (ROOT / "data/processed/auditoria.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle, report, histories, membership, predictions = run_experiments(frame)
    save_outputs(bundle, report, histories, membership, predictions)
    make_figures(frame, report, histories, ROOT / "reports/figures")
    from scripts.build_web import build_web
    build_web()
    print("Modelo selecionado:", report["selected"])
    print("Teste final:", report["test"])
