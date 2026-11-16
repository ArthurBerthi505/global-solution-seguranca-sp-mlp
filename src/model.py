"""MLP binária real: treinamento, validação, seleção e exportação ao navegador."""
from copy import deepcopy
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix,
                             classification_report, balanced_accuracy_score)
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src.data import ROOT, FEATURES, SEED, split_data

CONFIGS = [
    {"name": "MLP A", "hidden_layer_sizes": (16,), "alpha": 0.0001,
     "reason": "Referência com uma camada oculta e baixa capacidade."},
    {"name": "MLP B", "hidden_layer_sizes": (32, 16), "alpha": 0.0001,
     "reason": "Mais capacidade para interações entre região e contexto."},
    {"name": "MLP C", "hidden_layer_sizes": (64, 32, 16), "alpha": 0.01,
     "reason": "Mais profundidade com penalização L2 maior para conter overfitting."},
]
MAX_EPOCHS = 80
PATIENCE = 12
MIN_DELTA = 0.0001


def metrics(y, probabilities):
    """Médias macro dão o mesmo peso a furto e roubo. Limiar fixo 0,5."""
    pred = (np.asarray(probabilities) >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1]))}


def train_one(config, X_train, y_train, X_val, y_val, verbose=True):
    """Uma partial_fit equivale a uma época, mantendo o estado Adam.

    A validação externa controla a parada e o checkpoint. Não usamos
    early_stopping=True, que criaria outra divisão interna de validação.
    """
    model = MLPClassifier(hidden_layer_sizes=config["hidden_layer_sizes"],
                          activation="relu", solver="adam", alpha=config["alpha"],
                          learning_rate_init=0.001, batch_size=128,
                          random_state=SEED, shuffle=True, early_stopping=False)
    history, best_model = [], None
    best_loss, stale, best_epoch = np.inf, 0, 0
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for epoch in range(1, MAX_EPOCHS + 1):
            model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
            train_prob = model.predict_proba(X_train)[:, 1]
            val_prob = model.predict_proba(X_val)[:, 1]
            tm, vm = metrics(y_train, train_prob), metrics(y_val, val_prob)
            history.append({"epoch": epoch, "train_accuracy": tm["accuracy"],
                            "val_accuracy": vm["accuracy"], "train_loss": tm["log_loss"],
                            "val_loss": vm["log_loss"], "val_f1_macro": vm["f1_macro"]})
            if vm["log_loss"] < best_loss - MIN_DELTA:
                best_loss, stale, best_epoch = vm["log_loss"], 0, epoch
                best_model = deepcopy(model)
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    result = {"name": config["name"], "hidden_layers": list(config["hidden_layer_sizes"]),
              "alpha": config["alpha"], "best_epoch": best_epoch,
              "epochs_run": len(history), "parameters": int(sum(w.size + b.size for w, b in zip(best_model.coefs_, best_model.intercepts_))),
              "seconds": round(time.perf_counter() - started, 3),
              "validation": metrics(y_val, best_model.predict_proba(X_val)[:, 1]),
              "training": metrics(y_train, best_model.predict_proba(X_train)[:, 1])}
    if verbose:
        print(f"{result['name']} {result['hidden_layers']} | época {best_epoch}/{len(history)} | F1 val = {result['validation']['f1_macro']:.4f}", flush=True)
    return best_model, history, result


def run_experiments(frame, verbose=True):
    train, val, test = split_data(frame)
    # SOMENTE treino define as categorias. Entradas binárias dispensam scaler.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    X_train = encoder.fit_transform(train[FEATURES])
    X_val = encoder.transform(val[FEATURES])
    models, histories, results = [], {}, []
    for config in CONFIGS:
        model, history, result = train_one(config, X_train, train.alvo.to_numpy(),
                                           X_val, val.alvo.to_numpy(), verbose=verbose)
        models.append(model)
        histories[config["name"]] = history
        results.append(result)
    # Seleção exclusivamente pela validação: maior macro F1, menor perda, menos parâmetros.
    best_index = sorted(range(len(results)), key=lambda i: (
        -results[i]["validation"]["f1_macro"], results[i]["validation"]["log_loss"], results[i]["parameters"]))[0]
    selected = models[best_index]
    # Primeira avaliação do teste: somente o modelo selecionado.
    X_test = encoder.transform(test[FEATURES])
    p_test = selected.predict_proba(X_test)[:, 1]
    predictions = (p_test >= 0.5).astype(int)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, train.alvo)
    baseline_val = baseline.predict_proba(X_val)[:, 1]
    baseline_test = baseline.predict_proba(X_test)[:, 1]
    report = {"selected": results[best_index]["name"], "input_dimension": int(X_train.shape[1]),
              "features": FEATURES, "categories": {c: a.tolist() for c, a in zip(FEATURES, encoder.categories_)},
              "configs": results, "test": metrics(test.alvo, p_test),
              "confusion_matrix": confusion_matrix(test.alvo, predictions, labels=[0, 1]).tolist(),
              "per_class_test": classification_report(test.alvo, predictions, labels=[0, 1], target_names=["Furto", "Roubo"], output_dict=True, zero_division=0),
              "baseline": {"strategy": "most_frequent", "validation": metrics(val.alvo, baseline_val), "test": metrics(test.alvo, baseline_test)},
              "splits": {name: {"rows": len(d), "class_counts": {str(k): int(v) for k, v in d.alvo.value_counts().items()}} for name, d in zip(["train", "validation", "test"], [train, val, test])},
              "training_config": {"activation": "relu", "output": "logistic/sigmoid", "loss": "binary_crossentropy", "optimizer": "adam", "learning_rate": 0.001, "batch_size": 128, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "seed": SEED, "threshold": 0.5, "checkpoint": "minimum validation log_loss", "architecture_selection": "maximum validation macro F1"}}
    bundle = {"encoder": encoder, "model": selected, "features": FEATURES,
              "selected": report["selected"], "classes": ["Furto", "Roubo"]}
    split_membership = pd.concat([d[["id_grupo"]].assign(split=name) for d, name in zip([train, val, test], ["train", "validation", "test"])])
    predictions_frame = test[["id_grupo", "alvo"]].copy()
    predictions_frame["prob_roubo"] = p_test
    predictions_frame["estimado"] = predictions
    return bundle, report, histories, split_membership, predictions_frame


def export_model(bundle, report, destination):
    """Exporta os pesos REAIS e o mesmo encoder. Saída JS igual à Python."""
    model, encoder = bundle["model"], bundle["encoder"]
    payload = {"format": "sklearn-mlp-binary-v1", "selected": bundle["selected"],
               "features": FEATURES, "categories": [a.tolist() for a in encoder.categories_],
               "weights": [w.tolist() for w in model.coefs_],
               "biases": [b.tolist() for b in model.intercepts_],
               "activation": "relu", "output_activation": "sigmoid",
               "classes": ["Furto", "Roubo"], "threshold": 0.5,
               "metrics": report["test"], "input_dimension": report["input_dimension"],
               "sample_size": sum(s["rows"] for s in report["splits"].values()),
               "year": 2016}
    Path(destination).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def save_outputs(bundle, report, histories, membership, predictions, output_root=ROOT):
    output_root = Path(output_root)
    for directory in ["models", "reports", "web"]:
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_root / "models/mlp_bundle.joblib")
    (output_root / "reports/metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "reports/histories.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    membership.to_csv(output_root / "reports/split_membership.csv", index=False)
    predictions.to_csv(output_root / "reports/test_predictions.csv", index=False)
    rows = [{"architecture": r["name"], "layers": str(r["hidden_layers"]), **r["validation"]} for r in report["configs"]]
    pd.DataFrame(rows).to_csv(output_root / "reports/comparison_validation.csv", index=False)
    export_model(bundle, report, output_root / "web/model.json")


if __name__ == "__main__":
    frame = pd.read_csv(ROOT / "data/processed/amostra_capital_2016.csv")
    bundle, report, histories, membership, predictions = run_experiments(frame)
    save_outputs(bundle, report, histories, membership, predictions)
    print(json.dumps({"selected": report["selected"], "test": report["test"]}, indent=2))
