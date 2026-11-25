"""Tune softmax with text on the five shared student folds.
"""

from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data_exploration.dataexploration import load_dataset
from models.softmax.compare_softmax_features import with_text_columns
from models.softmax.softmax_regression import CLASSES, SoftmaxRegression, prepare_fold


LEARNING_RATES = (0.001, 0.01, 0.1)
BATCH_SIZES = (16, 32, 64)
L2_STRENGTHS = (0.0001, 0.001, 0.01)
SETTINGS = ["learning_rate", "batch_size", "l2"]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "softmax"


def tune_softmax(data):
    """Return the fold scores and the ranked summary for the proposal grid."""
    grid = list(product(LEARNING_RATES, BATCH_SIZES, L2_STRENGTHS))
    rows = []
    for fold_number in range(1, 6):
        X_train, X_val, y_train, y_val = prepare_fold(data, fold_number)
        columns = with_text_columns(X_train.columns)
        train_features = X_train.loc[:, columns].to_numpy()
        val_features = X_val.loc[:, columns].to_numpy()

        for learning_rate, batch_size, l2 in grid:
            model = SoftmaxRegression(learning_rate=learning_rate, batch_size=batch_size, l2=l2, seed=42)
            model.fit(train_features, y_train)
            print("Stop reason:", model.stop_reason_)
            print("Loss at last epoch:", model.loss_history_[-1])
            predictions = model.predict(val_features)
            rows.append({
                "fold": fold_number,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "l2": l2,
                "feature_columns": len(columns),
                "epochs": model.n_epochs_,
                "best_epoch": model.best_epoch_,
                "validation_accuracy": accuracy_score(y_val, predictions),
                "validation_macro_f1": f1_score(
                    y_val, predictions, labels=range(len(CLASSES)), average="macro",
                ),
            })
        print(f"Fold {fold_number}/5 finished ({len(rows)}/{5 * len(grid)} fits)", flush=True)

    fold_scores = pd.DataFrame(rows)
    summary = fold_scores.groupby(SETTINGS, as_index=False).agg(
        mean_macro_f1=("validation_macro_f1", "mean"),
        std_macro_f1=("validation_macro_f1", "std"),
        mean_accuracy=("validation_accuracy", "mean"),
        std_accuracy=("validation_accuracy", "std"),
    )
    summary = summary.sort_values(
        ["mean_macro_f1", "mean_accuracy"], ascending=False,
    ).reset_index(drop=True)
    return fold_scores, summary


if __name__ == "__main__":
    fold_scores, summary = tune_softmax(load_dataset())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(RESULTS_DIR / "tuning_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "tuning_summary.csv", index=False)
    print("\nTop five settings by mean validation macro F1")
    print(summary.head().round(6).to_string(index=False))
    print(f"\nSaved fold scores and ranked summary to {RESULTS_DIR}")
