"""Evaluate the current submit-package Gini model on the original split.

Run: python models/random_forest/evaluate_selected_model.py
Fit only the original 435 training rows, with parameters fixed before scoring.
The original test partition was exposed in earlier experiments; this is a
retrospective evaluation, not a new untouched-test estimate.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.data_preprocessing import DATA_FILE, LABELS, load_preprocessed_dataset, sha256
from models.random_forest.submit.random_forest import PARAMETERS_FILE, RandomForestModel


def main():
    saved, preprocessing = load_preprocessed_dataset()
    raw = pd.read_csv(DATA_FILE)
    if (raw.index.tolist() != saved.index.tolist() or raw.id.tolist() != saved.id.tolist()
            or raw.label.tolist() != saved.label.tolist()):
        raise ValueError("Raw responses do not align with the saved split.")
    if saved.groupby("id")["split"].nunique().ne(1).any():
        raise ValueError("Student overlap between original splits.")
    train_mask = saved.split.eq("train")
    model = RandomForestModel().fit(raw.loc[train_mask])
    if model.preprocessing_state_ != preprocessing["state"]:
        raise ValueError("Selected-package preprocessing no longer reproduces the original training state.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = PROJECT_ROOT / "results" / f"random_forest_selected_evaluation_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for split in ["train", "validation", "test"]:
        frame = raw.loc[saved.split.eq(split)]
        prediction = model.predict(frame.drop(columns=["id", "label"]))
        report = classification_report(frame.label, prediction, labels=LABELS, output_dict=True, zero_division=0)
        rows.append({
            "model": "selected_tuned_gini", "split": split,
            "rows": len(frame), "students": frame.id.nunique(),
            "correct": int(np.sum(frame.label.to_numpy() == np.asarray(prediction))),
            "accuracy": accuracy_score(frame.label, prediction),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        })
        pd.DataFrame({"source_row_index": frame.index, "actual": frame.label.to_numpy(),
                      "predicted": prediction}).to_csv(output / f"{split}_predictions.csv", index=False)
        pd.DataFrame(report).T.to_csv(output / f"{split}_classification_report.csv")
        pd.DataFrame(confusion_matrix(frame.label, prediction, labels=LABELS),
                     index=pd.Index(LABELS, name="actual"),
                     columns=pd.Index(LABELS, name="predicted")).to_csv(output / f"{split}_confusion_matrix.csv")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    config = {
        "model_source": "models/random_forest/submit/random_forest.py",
        "package_manifest_sha256": sha256(PARAMETERS_FILE.with_name("package_manifest.json")),
        "parameters_file": str(PARAMETERS_FILE.relative_to(PROJECT_ROOT)),
        "parameters_sha256": sha256(PARAMETERS_FILE), "model_options": model.model_options,
        "raw_data_sha256": sha256(DATA_FILE), "split": preprocessing["split"],
        "training_splits": ["train"], "training_rows": int(train_mask.sum()),
        "preprocessing_fitted_on": "original training rows only; matches frozen historical state",
        "tuned_in_this_run": False, "selected_before_test": True,
        "test_evaluated": True, "local_test_previously_exposed": True,
        "interpretation": "Original local test partition reused for retrospective reporting. Earlier baseline evaluation and all-data training-size experiments exposed these rows. Not a new untouched test or the hidden course test set.",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output / "evaluation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results: {output}")
    return output


if __name__ == "__main__":
    main()
