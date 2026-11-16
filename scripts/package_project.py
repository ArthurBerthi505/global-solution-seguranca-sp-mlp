"""Cria pacote público com código, modelo e resultados agregados.

Os registros e resultados por ocorrência são reproduzidos localmente.
"""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RECORD_FILES = {
    "data/processed/amostra_capital_2016.csv",
    "reports/split_membership.csv",
    "reports/test_predictions.csv",
}


def package(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".openai", ".venv", "__pycache__", ".ipynb_checkpoints", "dist", "node_modules"}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.resolve() == destination:
                continue
            rel = path.relative_to(ROOT)
            if excluded.intersection(rel.parts) or rel.parts[:2] == ("data", "raw"):
                continue
            if rel.as_posix() in LOCAL_RECORD_FILES:
                continue
            if rel.as_posix() in {"web/projeto.zip", "web/relatorio.pdf"} or path.suffix == ".pyc":
                continue
            archive.write(path, Path("GlobalSolution_SegurancaSP") / rel)
    print(destination)
    print("Tamanho:", destination.stat().st_size, "bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    package(parser.parse_args().destination)
