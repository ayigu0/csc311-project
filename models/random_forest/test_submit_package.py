"""Verify the selected source package independently of repository imports."""

import ast
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from data_exploration.dataexploration import DATA_FILE
from models.random_forest.submit import data_preprocessing as isolated

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = HERE / "archive"


def load_historical_dataset():
    return (pd.read_csv(ARCHIVE / "data/rf_features_v1.csv"),
            json.loads((ARCHIVE / "data/rf_preprocessing_v1.json").read_text()))


class SubmitPackageTests(unittest.TestCase):
    def test_rules_parameters_and_features_match_selected_version(self):
        archived_tree = ast.parse((ARCHIVE / "data_preprocessing.py").read_text())
        functions = {node.name: node for node in archived_tree.body if isinstance(node, ast.FunctionDef)}
        for name in ["normalize_text", "is_unknown", "clean_answers", "parse_ticket_price",
                     "tokens", "base_features", "fit_preprocessor", "transform_features"]:
            self.assertEqual(ast.dump(functions[name]),
                             ast.dump(ast.parse(inspect.getsource(getattr(isolated, name))).body[0]))
        self.assertEqual((ARCHIVE / "tuned_params.json").read_bytes(),
                         (HERE / "submit/tuned_params.json").read_bytes())
        saved, metadata = load_historical_dataset()
        raw = pd.read_csv(DATA_FILE)
        train = raw.loc[saved.split.eq("train")]
        state = isolated.fit_preprocessor(train)
        self.assertEqual(state, metadata["state"])
        self.assertEqual(state, json.loads((HERE / "submit/preprocessing_state.json").read_text()))
        generated = isolated.transform_features(raw, state)
        pd.testing.assert_frame_equal(saved[state["feature_columns"]], generated, check_names=False)
        self.assertEqual(saved.split.value_counts().to_dict(),
                         {"train": 435, "test": 261, "validation": 174})

    def test_standalone_copy_reproduces_validation_predictions(self):
        saved, _ = load_historical_dataset()
        raw = pd.read_csv(DATA_FILE)
        expected = pd.read_csv(ROOT / "results/random_forest/archive/random_forest_gini_20261124T004402_028162Z/validation_predictions.csv")
        with tempfile.TemporaryDirectory(prefix="rf-submit-test-") as folder:
            temp = Path(folder)
            shutil.copytree(HERE / "submit", temp / "submit", ignore=shutil.ignore_patterns("__pycache__"))
            raw.loc[saved.split.eq("train")].to_csv(temp / "train.csv", index=False)
            raw.loc[saved.split.eq("validation")].drop(columns=["id", "label"]).to_csv(temp / "validation.csv", index=False)
            script = """
import json, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'submit'))
from random_forest import RandomForestModel
model = RandomForestModel().fit(pd.read_csv(root / 'train.csv'))
predictions = model.predict(pd.read_csv(root / 'validation.csv'))
assert not any(name.startswith(('models.', 'data_exploration')) for name in sys.modules)
(root / 'predictions.json').write_text(json.dumps(predictions))
"""
            subprocess.run([sys.executable, "-I", "-c", script, str(temp)], cwd=temp, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((temp / "predictions.json").read_text()), expected.predicted.tolist())


if __name__ == "__main__":
    unittest.main()
