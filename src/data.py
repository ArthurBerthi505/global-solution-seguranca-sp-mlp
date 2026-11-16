"""Aquisição e recorte auditável de boletins históricos RDO/SSP-SP.

A localização do FATO vem da circunscrição, nunca de CIDADE (registro).
Uma linha final representa um boletim, não um objeto ou uma pessoa.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SOURCE = "https://www.kaggle.com/datasets/inquisitivecrow/crime-data-in-brazil"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/inquisitivecrow/crime-data-in-brazil/BO_2016.csv"
FEATURES = ["regiao", "contexto"]
KEY = ["ID_DELEGACIA", "ANO_BO", "NUM_BO"]
REGIONS = {
    "DEL.SEC.1º CENTRO": "Centro",
    "DEL.SEC.2º SUL": "Sul",
    "DEL.SEC.3º OESTE": "Oeste",
    "DEL.SEC.4º NORTE": "Norte",
    "DEL.SEC.5º LESTE": "Leste",
    "DEL.SEC.6º SANTO AMARO": "Santo Amaro",
    "DEL.SEC.7º ITAQUERA": "Itaquera",
    "DEL.SEC.8º SAO MATEUS": "São Mateus",
}
# Agrupamento por ambiente/contexto. Nenhuma categoria contém o rótulo.
# "Saidinha de Banco" foi excluída: descreve uma dinâmica ligada ao roubo.
CONTEXTS = {
    "TRANSEUNTE": "Transeunte",
    "VEICULO": "Veículo",
    "INTERIOR DE VEICULO": "Interior de veículo",
    "INTERIOR TRANSPORTE COLETIVO": "Transporte coletivo",
    "COLETIVO": "Transporte coletivo",
    "RESIDENCIA": "Residência",
    "CONDOMINIO RESIDENCIAL": "Residência",
    "ESTABELECIMENTO COMERCIAL": "Comércio",
    "INTERIOR ESTABELECIMENTO": "Comércio",
    "CONDOMINIO COMERCIAL": "Comércio",
    "JOALHERIA": "Comércio",
    "ESTABELECIMENTO ENSINO": "Ensino",
    "ESTABELECIMENTO BANCARIO": "Banco ou caixa eletrônico",
    "Caixa Eletrônico": "Banco ou caixa eletrônico",
    "CARGA": "Carga",
    "ESTABELECIMENTO-OUTROS": "Outros",
    "OUTROS": "Outros",
}
LABELS = {"Furto (art. 155)": 0, "Roubo (art. 157)": 1}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_raw(destination=None):
    destination = Path(destination or ROOT / "data/raw/BO_2016.csv.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(DOWNLOAD, destination)
    with zipfile.ZipFile(destination) as z:
        if "BO_2016.csv" not in z.namelist():
            raise ValueError("O download não contém BO_2016.csv.")
    return destination


def prepare_data(raw_path=None, sample_size=20_000):
    """Filtra, elimina ambiguidades e amostra preservando proporções das classes."""
    path = Path(raw_path or ROOT / "data/raw/BO_2016.csv.zip")
    cols = KEY + ["NOME_DEPARTAMENTO_CIRC", "NOME_SECCIONAL_CIRC",
                  "FLAG_STATUS", "RUBRICA", "CONDUTA"]
    with zipfile.ZipFile(path) as z:
        frame = pd.read_csv(z.open("BO_2016.csv"), usecols=cols,
                            dtype=str, encoding="utf-8-sig")
    for col in frame:
        frame[col] = frame[col].str.strip()
    audit = {"source": SOURCE, "download": DOWNLOAD,
             "file": "BO_2016.csv", "year": 2016,
             "raw_sha256": sha256(path), "raw_rows": len(frame),
             "geography": "DECAP + oito seccionais de circunscrição da capital",
             "features": FEATURES, "seed": SEED}

    # CIDADE indica local de REGISTRO no dicionário. Não filtrar por ela.
    geo = frame.NOME_DEPARTAMENTO_CIRC.eq("DECAP") & frame.NOME_SECCIONAL_CIRC.isin(REGIONS)
    frame = frame.loc[geo].copy()
    audit["capital_rows"] = len(frame)
    # O ano do boletim integra a chave; não entra no modelo.
    valid_key = frame[KEY].notna().all(axis=1) & frame[KEY].ne("").all(axis=1)
    frame = frame.loc[valid_key].copy()
    audit["valid_key_rows"] = len(frame)

    # Primeiro examinar TODAS as naturezas do BO, antes de filtrar o alvo.
    group = frame.groupby(KEY, sort=True, dropna=False)
    natures = group.RUBRICA.nunique(dropna=False)
    single_nature_keys = natures[natures.eq(1)].index
    audit["capital_unique_bulletins"] = len(natures)
    audit["bulletins_multiple_natures_excluded"] = int(natures.gt(1).sum())
    frame = frame.set_index(KEY).loc[lambda d: d.index.isin(single_nature_keys)].reset_index()
    frame = frame.loc[frame.RUBRICA.isin(LABELS) & frame.FLAG_STATUS.eq("C")].copy()
    audit["binary_consumated_rows"] = len(frame)
    audit["label_proxy_rows_excluded"] = int(frame.CONDUTA.eq("Saidinha de Banco").sum())
    frame = frame.loc[~frame.CONDUTA.eq("Saidinha de Banco")].copy()
    frame["regiao"] = frame.NOME_SECCIONAL_CIRC.map(REGIONS)
    # Ausência de contexto será imputada com categoria explícita.
    unmapped = frame.loc[frame.CONDUTA.notna() & ~frame.CONDUTA.isin(CONTEXTS), "CONDUTA"].unique().tolist()
    if unmapped:
        raise ValueError(f"Condutas não revisadas: {unmapped}")
    frame["contexto"] = frame.CONDUTA.map(CONTEXTS).fillna("Não informado")
    frame["alvo"] = frame.RUBRICA.map(LABELS).astype(int)

    # Um BO pode ter várias pessoas/objetos. Conflito de contexto ou região:
    # excluir o BO inteiro, sem escolher uma linha arbitrária.
    consistency = frame.groupby(KEY)[FEATURES + ["alvo"]].nunique(dropna=False)
    consistent = consistency.le(1).all(axis=1)
    audit["bulletins_conflicting_features_excluded"] = int((~consistent).sum())
    frame = frame.set_index(KEY).loc[lambda d: d.index.isin(consistency.index[consistent])].reset_index()
    before = len(frame)
    frame = frame.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    audit["duplicate_rows_collapsed"] = before - len(frame)
    audit["eligible_unique_bulletins"] = len(frame)
    audit["eligible_class_counts"] = {str(k): int(v) for k, v in frame.alvo.value_counts().items()}
    if len(frame) > sample_size:
        frame, _ = train_test_split(frame, train_size=sample_size,
                                    stratify=frame.alvo, random_state=SEED)
    frame = frame.sort_values(KEY).reset_index(drop=True)
    # Apenas hash não reversível da chave para auditoria de conjuntos.
    frame["id_grupo"] = frame[KEY].astype(str).agg("|".join, axis=1).map(
        lambda s: hashlib.sha256(s.encode()).hexdigest())
    frame = frame[["id_grupo"] + FEATURES + ["alvo"]]
    audit["sample_rows"] = len(frame)
    audit["sample_class_counts"] = {str(k): int(v) for k, v in frame.alvo.value_counts().items()}
    audit["missing_context_sample"] = int(frame.contexto.eq("Não informado").sum())
    return frame, audit


def split_data(frame):
    """Divisão estratificada 70/15/15 após consolidar uma linha por BO."""
    train, rest = train_test_split(frame, test_size=0.30, random_state=SEED,
                                   stratify=frame.alvo)
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED,
                                 stratify=rest.alvo)
    sets = [set(d.id_grupo) for d in [train, val, test]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return train.copy(), val.copy(), test.copy()


if __name__ == "__main__":
    data, audit = prepare_data(download_raw())
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "amostra_capital_2016.csv", index=False)
    (output / "auditoria.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
