"""Bounded two-stage RF search using the fixed validation split, never test.

Run: python models/random_forest/tune_random_forest.py
Stage 1: 18 structural configurations per criterion at 100 trees / sqrt.
Stage 2: vary feature sampling and tree count for each criterion's stage-1 winner.
This is a holdout search, not cross-validation or an exhaustive grid search.
"""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.data_preprocessing import (
    FEATURE_FILE, METADATA_FILE, LABELS, load_preprocessed_dataset, sha256,
)
from models.random_forest.random_forest import build_pipeline

TUNED_CONFIG = Path(__file__).resolve().parent / "tuned_params.json"
BASE_OPTIONS = dict(n_estimators=100, max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, max_features="sqrt", seed=42)


def rank(trial):
    # Among identical validation scores prefer fewer trees, then smaller trees.
    return (trial["validation_macro_f1"], trial["validation_accuracy"],
            -trial["n_estimators"], -trial["mean_tree_nodes"], -trial["trial_id"])


def main():
    data, preprocessing = load_preprocessed_dataset()
    columns = preprocessing["state"]["feature_columns"]
    if data.groupby("id")["split"].nunique().ne(1).any():
        raise ValueError("Student overlap between splits.")
    train = data.loc[data["split"].eq("train")]
    val = data.loc[data["split"].eq("validation")]
    if train.empty or val.empty or set(columns) & {"id", "label", "split", "source_row_index"}:
        raise ValueError("Invalid training/validation inputs.")
    X_train, y_train = train[columns], train["label"]
    X_val, y_val = val[columns], val["label"]
    # Drop the full table reference; the search never extracts test features/labels.
    del data
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = PROJECT_ROOT / "results" / f"random_forest_tuning_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    grid = list(product([5, 10, 20, None], [1, 2, 5], [2, 5, 10]))
    baseline_structure = (None, 1, 2)
    grid.remove(baseline_structure)
    structures = [baseline_structure, *random.Random(42).sample(grid, 17)]
    trials, options_by_id, predictions_by_id, seen = [], {}, {}, set()

    def try_options(stage, options):
        key = json.dumps(options, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        model = build_pipeline(**options)
        model.fit(X_train, y_train)
        training_pred, prediction = model.predict(X_train), model.predict(X_val)
        trial_id = len(trials) + 1
        row = {
            "trial_id": trial_id, "stage": stage, **options,
            "train_accuracy": accuracy_score(y_train, training_pred),
            "train_macro_f1": f1_score(y_train, training_pred, labels=LABELS, average="macro", zero_division=0),
            "validation_accuracy": accuracy_score(y_val, prediction),
            "validation_macro_f1": f1_score(y_val, prediction, labels=LABELS, average="macro", zero_division=0),
            "mean_tree_nodes": float(np.mean([tree.tree_.node_count for tree in model.named_steps["forest"].estimators_])),
        }
        row["accuracy_gap"] = row["train_accuracy"] - row["validation_accuracy"]
        trials.append(row)
        options_by_id[trial_id] = options.copy()
        predictions_by_id[trial_id] = prediction
        pd.DataFrame(trials).to_csv(run_dir / "trials.csv", index=False)
        if trial_id % 6 == 0:
            print(f"Finished {trial_id} configurations; best validation macro-F1: {max(trials, key=rank)['validation_macro_f1']:.4f}", flush=True)

    for criterion in ["gini", "entropy"]:
        for depth, leaf, split in structures:
            try_options("structure", {**BASE_OPTIONS, "criterion": criterion,
                                      "max_depth": depth, "min_samples_leaf": leaf,
                                      "min_samples_split": split})
    structural_winners = {
        criterion: max((row for row in trials if row["criterion"] == criterion), key=rank)
        for criterion in ["gini", "entropy"]
    }
    for criterion, winner in structural_winners.items():
        for count, features in product([100, 300], ["sqrt", "log2", 0.3]):
            try_options("features_and_trees", {
                **options_by_id[winner["trial_id"]], "n_estimators": count, "max_features": features,
            })
    best = max(trials, key=rank)
    summary = []
    for criterion in ["gini", "entropy"]:
        baseline = next(row for row in trials if options_by_id[row["trial_id"]] == {**BASE_OPTIONS, "criterion": criterion})
        winner = max((row for row in trials if row["criterion"] == criterion), key=rank)
        for label, row in [("baseline", baseline), ("tuned", winner)]:
            summary.append({"configuration": label, **row})
    pd.DataFrame(summary).to_csv(run_dir / "baseline_vs_tuned.csv", index=False)
    pd.concat([
        pd.DataFrame({"trial_id": trial_id, "source_row_index": val.index,
                      "actual": y_val.to_numpy(), "predicted": prediction})
        for trial_id, prediction in predictions_by_id.items()
    ], ignore_index=True).to_csv(run_dir / "all_validation_predictions.csv", index=False)
    chosen_predictions = predictions_by_id[best["trial_id"]]
    pd.DataFrame(
        confusion_matrix(y_val, chosen_predictions, labels=LABELS),
        index=pd.Index(LABELS, name="actual"), columns=pd.Index(LABELS, name="predicted"),
    ).to_csv(run_dir / "best_validation_confusion_matrix.csv")
    pd.DataFrame(classification_report(
        y_val, chosen_predictions, labels=LABELS, output_dict=True, zero_division=0,
    )).T.to_csv(run_dir / "best_validation_classification_report.csv")

    config = {
        "model_options": options_by_id[best["trial_id"]],
        "selected_trial": best,
        "feature_set": preprocessing["version"],
        "data_sha256": sha256(FEATURE_FILE),
        "preprocessing_metadata_sha256": sha256(METADATA_FILE),
        "split": preprocessing["split"],
        "test_evaluated": False,
        "search": {
            "method": "two-stage holdout search; same structural samples for both criteria",
            "structural_candidates_per_criterion": structures,
            "stage_2": {"n_estimators": [100, 300], "max_features": ["sqrt", "log2", 0.3]},
            "seed": 42, "unique_trials": len(trials),
            "selection": "validation macro-F1, then accuracy; ties prefer fewer trees, fewer mean nodes, earlier trial",
            "limitation": "selection score on one reused validation split; not an unbiased generalization estimate",
        },
        "results_directory": str(run_dir.relative_to(PROJECT_ROOT)),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    encoded = json.dumps(config, indent=2) + "\n"
    (run_dir / "best_params.json").write_text(encoded, encoding="utf-8")
    TUNED_CONFIG.write_text(encoded, encoding="utf-8")
    display = pd.DataFrame(summary)[["configuration", "criterion", "train_accuracy", "validation_accuracy", "validation_macro_f1"]]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected parameters: {config['model_options']}")
    print(f"Completed {len(trials)} trials. Test set not evaluated.")
    print(f"Results: {run_dir}\nReusable configuration: {TUNED_CONFIG}")
    return config


if __name__ == "__main__":
    main()
