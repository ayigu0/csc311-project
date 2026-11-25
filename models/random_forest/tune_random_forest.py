"""Retune Random Forest and compare validation scores on the shared development folds."""

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_exploration.dataexploration import DATA_FILE, split_by_student
from models.random_forest.submit.data_preprocessing import fit_preprocessor, transform_features

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "random_forest"
RESULTS_DIR = RESULTS_ROOT / "retuned_shared_cv"
REFERENCE_DIR = RESULTS_ROOT / "current" / "shared_cv"
PREVIOUS_PARAMS = Path(__file__).with_name("submit") / "tuned_params.json"
CLASSES = ["Baseball", "Basketball", "Soccer"]


def prepare_development_folds():
    """Load development answers only and fit the Random Forest features per fold."""
    ids = pd.read_csv(DATA_FILE, usecols=["id"], keep_default_na=False)["id"]
    reserved, folds = split_by_student(ids)
    # CSV record zero is the header; the shared split uses zero-based data indices.
    development = pd.read_csv(DATA_FILE, skiprows=(ids.index[reserved] + 1).tolist())
    development_ids = ids.loc[~reserved]
    if development["id"].tolist() != development_ids.tolist():
        raise ValueError("Loaded development rows do not match the shared student split.")
    development.index = development_ids.index

    encoded_folds = []
    for train_mask, validation_mask in folds:
        train = development.loc[ids.index[train_mask]]
        validation = development.loc[ids.index[validation_mask]]
        if not set(train["id"]).isdisjoint(validation["id"]):
            raise ValueError("Students overlap between training and validation.")
        train_answers = train.drop(columns=["id", "label"])
        validation_answers = validation.drop(columns=["id", "label"])
        preprocessing = fit_preprocessor(train_answers)
        encoded_folds.append((
            transform_features(train_answers, preprocessing),
            train["label"],
            transform_features(validation_answers, preprocessing),
            validation["label"],
        ))
    return encoded_folds


def score_predictions(actual, predicted):
    return {
        "validation_accuracy": accuracy_score(actual, predicted),
        "validation_macro_f1": f1_score(
            actual, predicted, labels=CLASSES, average="macro", zero_division=0,
        ),
    }


def summarize_scores(scores):
    return {
        "mean_macro_f1": scores["validation_macro_f1"].mean(),
        "std_macro_f1": scores["validation_macro_f1"].std(),
        "mean_accuracy": scores["validation_accuracy"].mean(),
        "std_accuracy": scores["validation_accuracy"].std(),
    }


def score_candidate(folds, options, trial_id, stage):
    """Refit the same candidate once per fold and collect validation predictions."""
    estimator_options = options.copy()
    estimator_options["random_state"] = estimator_options.pop("seed")
    rows, predictions = [], []
    for fold, (X_train, y_train, X_validation, actual) in enumerate(folds, 1):
        model = RandomForestClassifier(**estimator_options, n_jobs=-1)
        model.fit(X_train, y_train)
        predicted = model.predict(X_validation)
        rows.append({
            "fold": fold, "trial_id": trial_id, "stage": stage, **options,
            "feature_columns": X_train.shape[1],
            **score_predictions(actual, predicted),
        })
        predictions.append(pd.DataFrame({
            "fold": fold, "actual": actual, "predicted": predicted,
        }))
    scores = pd.DataFrame(rows)
    out_of_fold = pd.concat(predictions, verify_integrity=True).sort_index()
    return scores, out_of_fold


def rank(trial):
    return (
        trial["mean_macro_f1"], trial["mean_accuracy"],
        -trial["n_estimators"], -trial["trial_id"],
    )


def tune(folds, previous):
    """Run the historical two-stage search with every candidate scored by five-fold CV."""
    old_options = previous["model_options"]
    search = previous["search"]
    trials, all_scores, predictions_by_trial, options_by_trial = [], [], {}, {}
    seen = set()

    def try_options(options, stage):
        key = tuple(sorted(options.items()))
        if key in seen:
            return
        seen.add(key)
        trial_id = len(trials) + 1
        scores, predictions = score_candidate(folds, options, trial_id, stage)
        trial = {"trial_id": trial_id, "stage": stage, **options, **summarize_scores(scores)}
        trials.append(trial)
        all_scores.append(scores)
        predictions_by_trial[trial_id] = predictions
        options_by_trial[trial_id] = options
        print(f"Trial {trial_id:2d} [{stage}]: mean validation macro F1 "
              f"{trial['mean_macro_f1']:.4f}", flush=True)

    criteria = ("gini", "entropy")
    for criterion in criteria:
        for depth, leaf, split in search["structural_candidates_per_criterion"]:
            try_options({
                **old_options, "criterion": criterion, "max_depth": depth,
                "min_samples_leaf": leaf, "min_samples_split": split,
                "n_estimators": 100, "max_features": "sqrt",
            }, "structure")

    stage_two = search["stage_2"]
    structural_winners = [
        max((trial for trial in trials if trial["criterion"] == criterion), key=rank)
        for criterion in criteria
    ]
    for winner in structural_winners:
        for count, features in product(stage_two["n_estimators"], stage_two["max_features"]):
            try_options({
                **options_by_trial[winner["trial_id"]],
                "n_estimators": count, "max_features": features,
            }, "features_and_trees")

    # Include the prior tuned configuration even if neither structural winner reproduces it.
    try_options(old_options, "previous_winner")
    ranked = sorted(trials, key=rank, reverse=True)
    best_id = ranked[0]["trial_id"]
    return (
        pd.concat(all_scores, ignore_index=True),
        pd.DataFrame(ranked),
        options_by_trial[best_id],
        predictions_by_trial[best_id],
    )


def compare_models(rf_predictions):
    """Recalculate all three validation scores using their saved fold predictions."""
    predictions = {"random_forest": rf_predictions}
    for name in ("softmax", "naive_bayes"):
        saved = pd.read_csv(
            REFERENCE_DIR / f"{name}_validation_predictions.csv",
            index_col="source_row_index",
        ).sort_index()
        # Every model must predict the exact same rows, with the same labels and folds.
        pd.testing.assert_frame_equal(
            saved[["fold", "actual"]], rf_predictions[["fold", "actual"]], check_names=False,
        )
        predictions[name] = saved

    summaries = []
    for name, frame in predictions.items():
        scores = pd.DataFrame([
            {"fold": fold, "validation_rows": len(rows), **score_predictions(rows.actual, rows.predicted)}
            for fold, rows in frame.groupby("fold")
        ])
        summary = summarize_scores(scores)
        pooled = score_predictions(frame.actual, frame.predicted)
        summaries.append({
            "model": name, "folds": len(scores), "development_rows": len(frame),
            "accuracy_mean": summary["mean_accuracy"], "accuracy_std": summary["std_accuracy"],
            "macro_f1_mean": summary["mean_macro_f1"], "macro_f1_std": summary["std_macro_f1"],
            "pooled_oof_accuracy": pooled["validation_accuracy"],
            "pooled_oof_macro_f1": pooled["validation_macro_f1"],
        })
        if name == "random_forest":
            rf_scores = scores.rename(columns={
                "validation_accuracy": "accuracy", "validation_macro_f1": "macro_f1",
            }).set_index("fold")
    return pd.DataFrame(summaries).sort_values("macro_f1_mean", ascending=False), rf_scores


def save_confusion_matrix(predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    counts = confusion_matrix(predictions.actual, predictions.predicted, labels=CLASSES)
    matrix = pd.DataFrame(counts, index=CLASSES, columns=CLASSES).rename_axis("actual_sport")
    matrix.to_csv(RESULTS_DIR / "validation_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay(counts, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set(
        xlabel="Predicted sport", ylabel="Actual sport",
        title=f"Random Forest validation confusion matrix (n={len(predictions)})",
    )
    fig.savefig(RESULTS_DIR / "validation_confusion_matrix.png", dpi=240)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    previous = json.loads(PREVIOUS_PARAMS.read_text(encoding="utf-8"))
    folds = prepare_development_folds()
    all_scores, summary, options, predictions = tune(folds, previous)
    comparison, validation_scores = compare_models(predictions)
    all_scores.to_csv(RESULTS_DIR / "tuning_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "tuning_summary.csv", index=False)
    validation_scores.to_csv(RESULTS_DIR / "validation_metrics_by_fold.csv")
    validation_scores[["accuracy", "macro_f1"]].agg(["mean", "std"]).rename_axis(
        "statistic",
    ).to_csv(RESULTS_DIR / "validation_metrics_summary.csv")
    predictions.to_csv(RESULTS_DIR / "validation_predictions.csv", index_label="row_index")
    save_confusion_matrix(predictions)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    config = {
        "model_options": options,
        "selected_trial": int(summary.iloc[0]["trial_id"]),
        "protocol": "Same five student-grouped development folds as Softmax and Naive Bayes",
        "development_rows": len(predictions),
        "validation_rows_by_fold": validation_scores["validation_rows"].tolist(),
        "test_evaluated": False,
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "search": {
            "structural_candidates_per_criterion": previous["search"]["structural_candidates_per_criterion"],
            "stage_2": previous["search"]["stage_2"],
            "includes_previous_winner": True,
            "unique_trials": len(summary),
            "selection": "Highest mean validation macro F1, then mean accuracy, then fewer trees, then earlier trial",
        },
    }
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nSelected Random Forest settings: {options}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
