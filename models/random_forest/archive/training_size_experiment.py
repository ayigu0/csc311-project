"""Repeated student-grouped 70/80/90/95% training-size experiment.

Run: python models/random_forest/training_size_experiment.py
All 870 raw rows are eligible. The old split and artifacts are preserved but
are not used for this experiment. Each training subset fits its own preprocessing.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.data_preprocessing import (
    DATA_FILE, FEATURE_FILE, METADATA_FILE, LABELS, fit_preprocessor,
    transform_features, load_preprocessed_dataset, sha256,
)
from models.random_forest.random_forest import build_pipeline

HERE = Path(__file__).resolve().parent
FRACTIONS = [0.70, 0.80, 0.90, 0.95]
SPLIT_SEEDS = list(range(42, 52))


def scores(actual, predicted):
    return {"accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)}


def plot_results(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    for ax, criterion in zip(axes, ["gini", "entropy"]):
        part = summary.loc[summary.criterion.eq(criterion)].sort_values("train_fraction")
        x = part.train_fraction.to_numpy() * 100
        for prefix, label, color, style in [
            ("train", "Training", "#53565A", "--"),
            ("validation", "All remaining validation students", "#0072B2", "-"),
            ("common_validation", "Same 15 validation students per seed", "#D55E00", ":"),
        ]:
            mean = part[f"{prefix}_accuracy_mean"].to_numpy()
            std = part[f"{prefix}_accuracy_std"].to_numpy()
            ax.plot(x, mean, marker="o", color=color, linestyle=style, label=label)
            if prefix == "validation":
                ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.13)
        ax.set(title=criterion.capitalize(), xlabel="Requested training share (%)", xticks=[70, 80, 90, 95])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Training size: 10 student-grouped splits, frozen RF settings\nShaded band: split-to-split standard deviation, not a confidence interval", fontsize=11)
    fig.savefig(output / "training_size_accuracy.png", dpi=180)
    plt.close(fig)


def main():
    # Validate provenance without using the old feature matrix or split for fitting.
    load_preprocessed_dataset()
    settings_path = HERE / "tuned_params.json"
    settings = json.loads(settings_path.read_text())
    if settings["data_sha256"] != sha256(FEATURE_FILE) or settings["preprocessing_metadata_sha256"] != sha256(METADATA_FILE):
        raise ValueError("Current settings do not match the recorded RF preprocessing version.")
    options = settings["model_options"]
    protected = [path for path in (PROJECT_ROOT / "results").rglob("*") if path.is_file()]
    protected += [settings_path, FEATURE_FILE, METADATA_FILE]
    original_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in protected}
    raw = pd.read_csv(DATA_FILE)
    if raw.id.isna().any() or not raw.label.isin(LABELS).all():
        raise ValueError("Invalid IDs or labels.")
    students = sorted(raw.id.unique())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = PROJECT_ROOT / "results" / f"random_forest_training_size_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "preprocessing").mkdir()
    (output / "prior_artifact_hashes.json").write_text(json.dumps(original_hashes, indent=2) + "\n")
    trials, assignments, predictions = [], [], []

    for split_seed in SPLIT_SEEDS:
        order = np.random.default_rng(split_seed).permutation(students)
        common_ids = set(order[int(max(FRACTIONS) * len(students)):])
        previous_training_ids = set()
        for fraction in FRACTIONS:
            count = int(fraction * len(students))
            training_ids, validation_ids = set(order[:count]), set(order[count:])
            if training_ids & validation_ids or not previous_training_ids <= training_ids or not common_ids <= validation_ids:
                raise ValueError("Student split overlap or nesting failure.")
            previous_training_ids = training_ids
            train_mask = raw.id.isin(training_ids)
            common_mask = raw.id.isin(common_ids)
            train, validation = raw.loc[train_mask], raw.loc[~train_mask]
            train_answers = train.drop(columns=["id", "label"])
            validation_answers = validation.drop(columns=["id", "label"])
            state = fit_preprocessor(train_answers)
            X_train, X_validation = transform_features(train_answers, state), transform_features(validation_answers, state)
            if not X_train.columns.equals(X_validation.columns) or not X_train.index.equals(train.index) or not X_validation.index.equals(validation.index):
                raise ValueError("Feature alignment failure.")
            state_name = f"seed_{split_seed}_train_{int(fraction * 100)}.json"
            (output / "preprocessing" / state_name).write_text(json.dumps(state, indent=2) + "\n")
            assignments.append(pd.DataFrame({
                "split_seed": split_seed, "train_fraction": fraction,
                "source_row_index": raw.index, "id": raw.id,
                "partition": np.where(train_mask, "train", "validation"),
                "common_validation": common_mask,
            }))
            for criterion in ["gini", "entropy"]:
                model = build_pipeline(**{**options, "criterion": criterion}).fit(X_train, train.label)
                train_pred, val_pred = model.predict(X_train), model.predict(X_validation)
                common_positions = validation.id.isin(common_ids).to_numpy()
                row = {
                    "criterion": criterion, "split_seed": split_seed, "train_fraction": fraction,
                    "actual_train_fraction": len(training_ids) / len(students),
                    "train_students": len(training_ids), "validation_students": len(validation_ids),
                    "train_rows": len(train), "validation_rows": len(validation),
                    "common_validation_rows": int(common_positions.sum()), "features": X_train.shape[1],
                    "preprocessing_file": f"preprocessing/{state_name}",
                }
                for prefix, actual, prediction in [
                    ("train", train.label, train_pred),
                    ("validation", validation.label, val_pred),
                    ("common_validation", validation.label.to_numpy()[common_positions], val_pred[common_positions]),
                ]:
                    row.update({f"{prefix}_{metric}": value for metric, value in scores(actual, prediction).items()})
                row["accuracy_gap"] = row["train_accuracy"] - row["validation_accuracy"]
                row["macro_f1_gap"] = row["train_macro_f1"] - row["validation_macro_f1"]
                trials.append(row)
                for partition, frame, prediction in [("train", train, train_pred), ("validation", validation, val_pred)]:
                    predictions.append(pd.DataFrame({
                        "criterion": criterion, "split_seed": split_seed, "train_fraction": fraction,
                        "partition": partition, "source_row_index": frame.index,
                        "common_validation": frame.id.isin(common_ids).to_numpy(),
                        "actual": frame.label.to_numpy(), "predicted": prediction,
                    }))
            pd.DataFrame(trials).to_csv(output / "trials.csv", index=False)
        print(f"Completed split seed {split_seed}: {len(trials)} / 80 model fits", flush=True)

    by_run = pd.DataFrame(trials)
    numeric_metrics = [f"{partition}_{metric}" for partition in ["train", "validation", "common_validation"]
                       for metric in ["accuracy", "macro_f1"]] + ["accuracy_gap", "macro_f1_gap"]
    summary_rows = []
    for (criterion, fraction), group in by_run.groupby(["criterion", "train_fraction"], sort=False):
        row = {"criterion": criterion, "train_fraction": fraction, "repeats": len(group)}
        for key in ["train_students", "validation_students", "train_rows", "validation_rows", "common_validation_rows"]:
            row[key] = int(group[key].iloc[0])
        for metric in numeric_metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(output / "split_assignments.csv.gz", index=False, compression="gzip")
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    for relative, expected in original_hashes.items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"A prior artifact changed: {relative}")
    metadata = {
        "experiment": "repeated student-grouped training-size experiment",
        "train_fractions_requested": FRACTIONS, "split_seeds": SPLIT_SEEDS,
        "students": len(students), "rows": len(raw), "raw_data_sha256": sha256(DATA_FILE),
        "preprocessing": "RF v1 rules, freshly fitted within every training subset",
        "preprocessing_code_sha256": sha256(HERE / "data_preprocessing.py"),
        "frozen_tuning_config_sha256": sha256(settings_path),
        "model_options_by_criterion": {criterion: {**options, "criterion": criterion} for criterion in ["gini", "entropy"]},
        "hyperparameter_policy": "same structural settings as previous 5-fold experiment; vary only criterion; no new tuning",
        "original_split_used": False, "independent_test_set": False,
        "old_test_rows_eligible": True, "previous_artifacts_preserved": True,
        "group_split_policy": "floor(fraction * student_count) training students; remaining students validation; same permutation across fractions within seed",
        "common_validation_policy": "last 15 students in each seed's permutation; same evaluation rows across all four training fractions within that seed",
        "uncertainty": "sample standard deviation across 10 overlapping random splits; not a confidence interval or independent experiments",
        "interpretation": "full validation composition changes with ratio; use common-cohort results as a size-effect diagnostic; all data were used in prior exploration/selection, so this is exploratory",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output / "experiment_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(summary, output)
    display = summary[["criterion", "train_fraction", "train_rows", "validation_rows", "train_accuracy_mean",
                       "validation_accuracy_mean", "validation_accuracy_std", "common_validation_accuracy_mean", "accuracy_gap_mean"]]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {output}")
    return output


if __name__ == "__main__":
    main()
