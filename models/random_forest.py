"""Random Forest baseline using the shared eight numeric features.

Run from the project root: python models/random_forest.py
This is development code, not the restricted-library submission pred.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.dataexploration import (
    DATA_FILE,
    load_dataset,
    make_features,
    split_by_student,
)


SPLIT_SEED = 42
VAL_FRAC = 0.2
TEST_FRAC = 0.3
LABELS = ["Soccer", "Baseball", "Basketball"]


def build_pipeline(n_estimators=100, max_depth=None, min_samples_leaf=1,
                   max_features="sqrt", seed=42):
    """Keep learned imputation inside the model's training pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("forest", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def run_baseline(output_root=None, **model_options):
    """Train once, evaluate train/validation, and save an isolated run folder.

    All rows are retained for this first baseline, including empty responses.
    Missing features use training medians (zero for an entirely missing training
    column). This is a provisional policy until shared preprocessing is settled.
    """
    data = load_dataset()
    train, val, test = split_by_student(
        data["id"], val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SPLIT_SEED,
    )
    # Split raw rows first; do not build features or evaluate on the test set.
    X_train = make_features(data.loc[train])
    X_val = make_features(data.loc[val])
    y_train = data.loc[train, "label"]
    y_val = data.loc[val, "label"]

    model = build_pipeline(**model_options)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    metrics = pd.DataFrame([
        {
            "split": name,
            "rows": len(target),
            "accuracy": accuracy_score(target, pred),
            "macro_f1": f1_score(
                target, pred, labels=LABELS, average="macro", zero_division=0,
            ),
        }
        for name, target, pred in [
            ("train", y_train, train_pred), ("validation", y_val, val_pred),
        ]
    ])
    matrix = pd.DataFrame(
        confusion_matrix(y_val, val_pred, labels=LABELS),
        index=pd.Index(LABELS, name="actual"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    report = classification_report(
        y_val, val_pred, labels=LABELS, output_dict=True, zero_division=0,
    )

    output_root = Path(output_root) if output_root is not None else PROJECT_ROOT / "results"
    run_dir = output_root / datetime.now(timezone.utc).strftime("random_forest_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    matrix.to_csv(run_dir / "validation_confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(run_dir / "validation_classification_report.csv")
    pd.DataFrame({
        "source_row_index": X_val.index,
        "actual": y_val.to_numpy(),
        "predicted": val_pred,
    }).to_csv(run_dir / "validation_predictions.csv", index=False)

    metadata = {
        "model": "RandomForestClassifier",
        "feature_set": "shared_baseline_8_numeric_features",
        "features": X_train.columns.tolist(),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "split": {
            "seed": SPLIT_SEED, "val_fraction": VAL_FRAC, "test_fraction": TEST_FRAC,
            "rows": {name: int(mask.sum()) for name, mask in
                     [("train", train), ("validation", val), ("test_reserved", test)]},
        },
        "preprocessing": {
            "imputation": "training median; all-missing training columns use zero",
            "row_policy": "retain all rows, including empty responses",
            "scaling": "none",
            "training_imputation_values": dict(zip(
                X_train.columns, model.named_steps["imputer"].statistics_.tolist(),
            )),
        },
        "hyperparameters": model.named_steps["forest"].get_params(),
        "tuned": False,
        "test_evaluated": False,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nValidation confusion matrix (rows = actual, columns = predicted):")
    print(matrix.to_string())
    print(f"\nTest set reserved: {int(test.sum())} rows; not evaluated.")
    print(f"Results saved to: {run_dir}")
    return model, metrics, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", choices=["sqrt", "log2"], default="sqrt")
    parser.add_argument("--seed", type=int, default=42, help="Forest seed; split seed stays 42.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    run_baseline(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
