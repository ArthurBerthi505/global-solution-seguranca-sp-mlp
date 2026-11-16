"""Verificações de integridade, isolamento dos conjuntos e integração real."""
import json
import subprocess
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES, split_data


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "data/processed/amostra_capital_2016.csv")
        cls.bundle = joblib.load(ROOT / "models/mlp_bundle.joblib")

    def test_no_target_or_identifier_among_inputs(self):
        self.assertEqual(FEATURES, ["regiao", "contexto"])
        self.assertEqual(self.frame.id_grupo.nunique(), len(self.frame))
        self.assertEqual(set(self.frame.alvo), {0, 1})

    def test_splits_are_disjoint_and_reproducible(self):
        a = split_data(self.frame)
        b = split_data(self.frame)
        self.assertEqual([len(d) for d in a], [14000, 3000, 3000])
        for x, y in zip(a, b):
            self.assertEqual(x.id_grupo.tolist(), y.id_grupo.tolist())
        sets = [set(d.id_grupo) for d in a]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_export_matches_python_in_all_supported_combinations(self):
        enc = self.bundle["encoder"]
        inputs = [{"regiao": r, "contexto": c} for r in enc.categories_[0] for c in enc.categories_[1]]
        # Encoder de produção mantém handle_unknown=ignore, como o Python.
        inputs += [{"regiao": "Categoria não observada", "contexto": "Categoria não observada"}]
        script = """
const fs=require('fs'),mlp=require('./web/inference.js');
const model=JSON.parse(fs.readFileSync('./web/model.json','utf8'));
const inputs=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(inputs.map(x=>mlp.predict(model,x).roubo)));
"""
        proc = subprocess.run(["node", "-e", script], cwd=ROOT, input=json.dumps(inputs),
                               text=True, capture_output=True, check=True)
        actual = np.array(json.loads(proc.stdout))
        matrix = enc.transform(pd.DataFrame(inputs)[FEATURES])
        expected = self.bundle["model"].predict_proba(matrix)[:,1]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    def test_saved_test_metrics_match_predictions(self):
        from src.model import metrics
        data = pd.read_csv(ROOT / "reports/test_predictions.csv")
        report = json.loads((ROOT / "reports/metrics.json").read_text())
        calculated = metrics(data.alvo, data.prob_roubo)
        for name, value in calculated.items():
            self.assertAlmostEqual(value, report["test"][name], places=12)


if __name__ == "__main__":
    unittest.main()
