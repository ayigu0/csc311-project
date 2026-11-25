"""Fit the selected RF on all shared development rows and score shared test.

Run: python models/random_forest/evaluate_shared_test.py
Use the teammates' exact 80/20 student partition. No model selection or tuning
occurs here. This reused local holdout is not a fresh independent blind test.
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

from data_exploration.dataexploration import DATA_FILE, load_dataset, split_by_student
from models.random_forest.reporting_utils import sha256
from models.random_forest.submit.random_forest import PARAMETERS_FILE, RandomForestModel

CLASSES = ["Baseball", "Basketball", "Soccer"]


def main():
    shared = load_dataset()
    raw = pd.read_csv(DATA_FILE)
    if shared.id.tolist() != raw.id.tolist() or shared.label.tolist() != raw.label.tolist():
        raise ValueError("RF rows do not align with the shared dataset.")
    test, _ = split_by_student(shared.id)
    training = ~test
    if set(raw.loc[training, "id"]) & set(raw.loc[test, "id"]):
        raise ValueError("Student overlap between development and test.")
    model = RandomForestModel().fit(raw.loc[training])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = PROJECT_ROOT / "results/random_forest/runs" / f"shared_test_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    assignments = raw[["id"]].copy()
    assignments["partition"] = np.where(test, "test", "development_train")
    assignments.to_csv(output / "split_assignments.csv", index_label="source_row_index")
    rows = []
    for split, mask in [("development_train", training), ("test", test)]:
        frame = raw.loc[mask]
        predicted = model.predict(frame.drop(columns=["id", "label"]))
        report = classification_report(frame.label, predicted, labels=CLASSES, output_dict=True, zero_division=0)
        rows.append({
            "model": "selected_gini", "split": split,
            "rows": len(frame), "students": frame.id.nunique(),
            "correct": int(np.sum(frame.label.to_numpy() == np.asarray(predicted))),
            "accuracy": accuracy_score(frame.label, predicted),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        })
        pd.DataFrame({"source_row_index": frame.index, "actual": frame.label.to_numpy(),
                      "predicted": predicted}).to_csv(output / f"{split}_predictions.csv", index=False)
        pd.DataFrame(report).T.to_csv(output / f"{split}_classification_report.csv")
        pd.DataFrame(confusion_matrix(frame.label, predicted, labels=CLASSES),
                     index=pd.Index(CLASSES, name="actual"),
                     columns=pd.Index(CLASSES, name="predicted")).to_csv(output / f"{split}_confusion_matrix.csv")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "preprocessing_state.json").write_text(json.dumps(model.preprocessing_state_, indent=2) + "\n")
    config = {
        "protocol": "same shared 20% student holdout as Softmax and Naive Bayes; fit all remaining development rows",
        "split_function": "data_exploration.dataexploration.split_by_student",
        "split_source_sha256": sha256(PROJECT_ROOT / "data_exploration/dataexploration.py"),
        "test_fraction": 0.2, "split_seed": 42,
        "training_rows": int(training.sum()), "test_rows": int(test.sum()),
        "training_students": raw.loc[training, "id"].nunique(), "test_students": raw.loc[test, "id"].nunique(),
        "training_population": "full development set, including all five CV folds",
        "model_options": model.model_options, "parameters_sha256": sha256(PARAMETERS_FILE),
        "package_manifest_sha256": sha256(PARAMETERS_FILE.with_name("package_manifest.json")),
        "raw_data_sha256": sha256(DATA_FILE),
        "preprocessing_fit_rows": model.preprocessing_state_["training_rows"],
        "feature_count": len(model.preprocessing_state_["feature_columns"]),
        "new_tuning": False, "test_evaluated": True, "test_previously_exposed": True,
        "limitation": "Shared local test membership was exposed in historical RF test/training-size experiments. This run excludes it from fitting, but it is not a new untouched test or the hidden course test set.",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output / "evaluation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results: {output}")
    return output


if __name__ == "__main__":
    main()
