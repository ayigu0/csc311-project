"""Random Forest using the versioned, preprocessed RF dataset.

Run from the project root: python models/random_forest/random_forest.py
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.data_preprocessing import (
    FEATURE_FILE,
    METADATA_FILE,
    LABELS,
    load_preprocessed_dataset,
)


def build_pipeline(n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                   max_features="sqrt", seed=42, criterion="gini"):
    """The versioned input already contains training-fitted preprocessing."""
    return Pipeline([
        ("forest", RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def run_baseline(output_root=None, tuning_source=None, **model_options):
    """Train once, evaluate train/validation, and save an isolated run folder.

    Read frozen features and split assignments from the RF CSV. No preprocessing
    is refitted here. Test rows remain reserved and are never scored.
    """
    data, preprocessing = load_preprocessed_dataset()
    train, val, test = (data["split"].eq(name) for name in ["train", "validation", "test"])
    feature_columns = preprocessing["state"]["feature_columns"]
    X_train = data.loc[train, feature_columns]
    X_val = data.loc[val, feature_columns]
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
    criterion = model.named_steps["forest"].criterion
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / f"random_forest_{criterion}_{timestamp}"
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
        "feature_set": preprocessing["version"],
        "features": X_train.columns.tolist(),
        "data_file": str(FEATURE_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": hashlib.sha256(FEATURE_FILE.read_bytes()).hexdigest(),
        "split": preprocessing["split"],
        "preprocessing": {
            "metadata_file": str(METADATA_FILE.relative_to(PROJECT_ROOT)),
            "metadata_sha256": hashlib.sha256(METADATA_FILE.read_bytes()).hexdigest(),
            "policies": preprocessing["policies"],
            "training_imputation_values": preprocessing["state"]["medians"],
        },
        "hyperparameters": model.named_steps["forest"].get_params(),
        "tuned": tuning_source is not None,
        "tuning_source": tuning_source,
        "test_evaluated": False,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(f"\nRandom Forest criterion: {criterion}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nValidation confusion matrix (rows = actual, columns = predicted):")
    print(matrix.to_string())
    print(f"\nTest set reserved: {int(test.sum())} rows; not evaluated.")
    print(f"Results saved to: {run_dir}")
    return model, metrics, run_dir


def parse_max_features(value):
    if value in {"sqrt", "log2"}:
        return value
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use sqrt, log2, or a fraction in (0, 1].") from exc
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("Feature fraction must be in (0, 1].")
    return fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Use a saved tuning configuration for this dataset.")
    parser.add_argument("--criterion", choices=["gini", "entropy"], default=argparse.SUPPRESS)
    parser.add_argument("--n-estimators", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-depth", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-samples-leaf", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-samples-split", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-features", type=parse_max_features, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Forest seed; split seed stays 42.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    options = vars(parser.parse_args())
    config_path = options.pop("config")
    if config_path is not None:
        if set(options) - {"output_root"}:
            parser.error("Use --config alone or explicit model flags, not both.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config["data_sha256"] != hashlib.sha256(FEATURE_FILE.read_bytes()).hexdigest()
                or config["preprocessing_metadata_sha256"] != hashlib.sha256(METADATA_FILE.read_bytes()).hexdigest()):
            parser.error("Tuned configuration belongs to a different dataset/preprocessing version.")
        options.update(config["model_options"])
        options["tuning_source"] = str(config_path.resolve())
    run_baseline(**options)


if __name__ == "__main__":
    main()
