"""Compare the performance for softmax with text features and without text features
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data_exploration.dataexploration import load_dataset
from data_exploration.preprocessing import TEXT
from models.softmax.softmax_regression import CLASSES, SoftmaxRegression, prepare_fold


STARTING_SETTINGS = {"learning_rate": 0.01, "batch_size": 32, "l2": 0.001, "seed": 42}
TEXT_PREFIXES = tuple(f"{field}__" for field in TEXT)
METRICS = ["validation_accuracy", "validation_macro_f1"]


def with_text_columns(columns):
    """Select the feature set used in the with-text comparison."""
    # This flag mixes structured and text answers, so neither run uses it.
    return columns.drop("all_answers_missing")


def compare_feature_sets(data):
    """Score both feature sets using identical folds and training settings."""
    rows = []
    for fold_number in range(1, 6):
        X_train, X_val, y_train, y_val = prepare_fold(data, fold_number)

        with_text = with_text_columns(X_train.columns)
        structured = with_text[~with_text.str.startswith(TEXT_PREFIXES)]

        for name, columns in (("structured", structured), ("with_text", with_text)):
            model = SoftmaxRegression(**STARTING_SETTINGS)
            model.fit(X_train.loc[:, columns], y_train)
            predictions = model.predict(X_val.loc[:, columns])
            rows.append({
                "fold": fold_number,
                "feature_set": name,
                "feature_columns": len(columns),
                "epochs": model.n_epochs_,
                "validation_accuracy": accuracy_score(y_val, predictions),
                "validation_macro_f1": f1_score(
                    y_val, predictions, labels=range(len(CLASSES)), average="macro",
                ),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = compare_feature_sets(load_dataset())
    summary = results.groupby("feature_set", sort=False)[METRICS].agg(["mean", "std"])
    print(results.round(4).to_string(index=False))
    print("\nFive-fold mean and standard deviation")
    print(summary.round(4).to_string())
