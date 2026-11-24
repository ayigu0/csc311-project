"""Evaluate fixed Gini/entropy forests on train, validation and local test.

Run: python models/random_forest/evaluate_random_forest.py
Both forests fit only training rows. Select the preferred criterion using
validation macro-F1 (then accuracy) before scoring either on the test set.
Use compare_random_forest.py for subsequent validation-only experiments.
"""

import argparse
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

from models.random_forest.data_preprocessing import (
    FEATURE_FILE, METADATA_FILE, LABELS, load_preprocessed_dataset, sha256,
)
from models.random_forest.random_forest import build_pipeline


def evaluate(output_root=None):
    data, preprocessing = load_preprocessed_dataset()
    features = preprocessing["state"]["feature_columns"]
    split_names = ["train", "validation", "test"]
    if data["id"].isna().any() or not data["label"].isin(LABELS).all():
        raise ValueError("Missing student IDs or invalid labels.")
    if data.index.has_duplicates or not data["split"].isin(split_names).all():
        raise ValueError("Invalid row identifiers or split assignments.")
    if data.groupby("id")["split"].nunique().ne(1).any():
        raise ValueError("A student's rows appear in more than one split.")
    splits = {name: data.loc[data["split"].eq(name)] for name in split_names}
    expected = preprocessing["split"]["rows"]
    for name, frame in splits.items():
        expected_rows = expected["test_reserved" if name == "test" else name]
        if frame.empty or len(frame) != expected_rows:
            raise ValueError(f"Unexpected row count for {name}.")

    output_root = Path(output_root) if output_root is not None else PROJECT_ROOT / "results"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / f"random_forest_evaluation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    rows, models = [], {}

    def score(criterion, split):
        frame = splits[split]
        target = frame["label"]
        prediction = models[criterion].predict(frame[features])
        report = classification_report(
            target, prediction, labels=LABELS, output_dict=True, zero_division=0,
        )
        row = {
            "criterion": criterion, "split": split,
            "rows": len(frame), "students": frame["id"].nunique(),
            "accuracy": accuracy_score(target, prediction),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        }
        rows.append(row)
        folder = run_dir / criterion
        folder.mkdir(exist_ok=True)
        pd.DataFrame(report).T.to_csv(folder / f"{split}_classification_report.csv")
        pd.DataFrame(
            confusion_matrix(target, prediction, labels=LABELS),
            index=pd.Index(LABELS, name="actual"),
            columns=pd.Index(LABELS, name="predicted"),
        ).to_csv(folder / f"{split}_confusion_matrix.csv")
        pd.DataFrame({
            "source_row_index": frame.index,
            "actual": target.to_numpy(), "predicted": prediction,
        }).to_csv(folder / f"{split}_predictions.csv", index=False)
        return row

    validation = {}
    for criterion in ["gini", "entropy"]:
        models[criterion] = build_pipeline(criterion=criterion)
        models[criterion].fit(splits["train"][features], splits["train"]["label"])
        score(criterion, "train")
        validation[criterion] = score(criterion, "validation")
    selected = max(validation, key=lambda key: (
        validation[key]["macro_f1"], validation[key]["accuracy"], key == "gini",
    ))
    # Criterion selection is frozen before accessing either test score.
    for criterion in models:
        score(criterion, "test")
    metrics = pd.DataFrame(rows)
    metrics["split"] = pd.Categorical(metrics["split"], categories=split_names, ordered=True)
    metrics = metrics.sort_values(["criterion", "split"]).reset_index(drop=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metadata = {
        "feature_set": preprocessing["version"], "features": features,
        "data_file": str(FEATURE_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": sha256(FEATURE_FILE),
        "preprocessing_metadata_sha256": sha256(METADATA_FILE),
        "split": preprocessing["split"],
        "students_per_split": {key: int(frame["id"].nunique()) for key, frame in splits.items()},
        "hyperparameters": {key: model.named_steps["forest"].get_params() for key, model in models.items()},
        "training_splits": ["train"], "tuned_in_this_run": False,
        "selection_rule": "validation macro-F1, then validation accuracy; exact ties prefer gini",
        "selected_criterion_before_test": selected,
        "test_evaluated": True,
        "test_scope": "local student holdout, not the course's unseen TA/instructor test set",
        "test_usage": "reporting only; do not use these scores to tune or select models",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (run_dir / "evaluation_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected using validation before test evaluation: {selected}")
    print("Local test has now been evaluated. Use validation for future tuning.")
    print(f"Results saved to: {run_dir}")
    return metrics, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    evaluate(**vars(parser.parse_args()))
