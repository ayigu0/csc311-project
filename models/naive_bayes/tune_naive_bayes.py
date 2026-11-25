"""Select Mixed Naive Bayes settings on the five development folds."""

from itertools import product
from pathlib import Path

import pandas as pd

from data_exploration.dataexploration import load_dataset, split_by_student
from models.naive_bayes.naive_bayes import (
    MixedNaiveBayes,
    prepare_fold,
    score_predictions,
)


MIN_FREQUENCIES = (2, 3, 5)
MAX_TERMS = (25, 50, 100, 200, 300)
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
TEXT_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
VAR_SMOOTHING = 0.01
SETTING_COLUMNS = [
    "min_text_frequency", "max_terms_per_field", "alpha", "text_weight", "var_smoothing",
]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "naive_bayes"


def tune_naive_bayes(data):
    _, folds = split_by_student(data["id"])
    rows = []
    for fold, (train_mask, validation_mask) in enumerate(folds, start=1):
        train, validation = data.loc[train_mask], data.loc[validation_mask]
        for min_frequency, max_terms in product(MIN_FREQUENCIES, MAX_TERMS):
            # These encoded matrices are reusable across alpha and text-weight values.
            X_train, X_validation = prepare_fold(train, validation, min_frequency, max_terms)
            for alpha, text_weight in product(ALPHAS, TEXT_WEIGHTS):
                model = MixedNaiveBayes(
                    alpha=alpha, text_weight=text_weight, var_smoothing=VAR_SMOOTHING,
                )
                predicted = model.fit(X_train, train["label"]).predict(X_validation)
                rows.append({
                    "fold": fold,
                    "min_text_frequency": min_frequency,
                    "max_terms_per_field": max_terms,
                    "alpha": alpha,
                    "text_weight": text_weight,
                    "var_smoothing": VAR_SMOOTHING,
                    "feature_columns": X_train.shape[1],
                    **score_predictions(validation["label"], predicted),
                })
        print(f"Finished fold {fold}/{len(folds)}", flush=True)

    fold_scores = pd.DataFrame(rows)
    summary = fold_scores.groupby(SETTING_COLUMNS, as_index=False).agg(
        mean_macro_f1=("validation_macro_f1", "mean"),
        std_macro_f1=("validation_macro_f1", "std"),
        mean_accuracy=("validation_accuracy", "mean"),
        std_accuracy=("validation_accuracy", "std"),
    )
    summary = summary.sort_values(
        ["mean_macro_f1", "mean_accuracy", "max_terms_per_field"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return fold_scores, summary


def main():
    fold_scores, summary = tune_naive_bayes(load_dataset())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(RESULTS_DIR / "tuning_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "tuning_summary.csv", index=False)
    print(summary.head().round(6).to_string(index=False))


if __name__ == "__main__":
    main()
