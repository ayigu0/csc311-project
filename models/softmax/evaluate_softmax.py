"""Build a confusion matrix from the selected softmax model's validation folds.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from data_exploration.dataexploration import load_dataset, split_by_student
from models.softmax.compare_softmax_features import with_text_columns
from models.softmax.softmax_regression import CLASSES, MAX_EPOCHS, SoftmaxRegression, prepare_fold


BEST_SETTINGS = {"learning_rate": 0.1, "batch_size": 32, "l2": 0.0001, "seed": 42}
CLASS_IDS = range(len(CLASSES))
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "softmax"


def collect_validation_predictions(data):
    """Collect validation predictions for the confusion matrix.

    The input is the labeled survey DataFrame with student IDs. For each of
    the five shared folds, fit a fresh preprocessor and softmax model on the
    other four folds. Use the winning hyperparameters and the same training
    seed for each fit, then predict the responses in that validation fold.

    Return a DataFrame with 'fold', 'actual', and 'predicted' columns,
    sorted by the original row index. The actual and predicted values are
    integer class IDs in 'CLASSES' order. Each development response appears
    exactly once, and the held-out test responses are excluded.
    """
    test_mask, folds = split_by_student(data["id"])
    predictions_by_fold = []
    for fold_number, (_, validation_mask) in enumerate(folds, start=1):
        X_train, X_val, y_train, y_val = prepare_fold(data, fold_number)
        columns = with_text_columns(X_train.columns)
        model = SoftmaxRegression(**BEST_SETTINGS).fit(X_train.loc[:, columns], y_train)
        predictions = model.predict(X_val.loc[:, columns])
        if not y_val.index.equals(data.index[validation_mask]):
            raise ValueError(f"Validation rows do not match the shared split for fold {fold_number}.")

        predictions_by_fold.append(pd.DataFrame({
            "fold": fold_number, "actual": y_val, "predicted": predictions,
        }))
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, labels=CLASS_IDS, average="macro")
        print(f"Fold {fold_number}: {len(y_val)} rows, {model.n_epochs_} epochs, "
              f"accuracy {accuracy:.6f}, macro F1 {macro_f1:.6f}")

    combined = pd.concat(predictions_by_fold).sort_index()
    development_rows = data.index[~test_mask].sort_values()
    if not combined.index.is_unique or not combined.index.equals(development_rows):
        raise ValueError("Validation predictions must cover each development row exactly once.")
    return combined


def main():
    print(f"Selected softmax settings: {BEST_SETTINGS}; epoch cap: {MAX_EPOCHS}")
    predictions = collect_validation_predictions(load_dataset())
    counts = confusion_matrix(predictions["actual"], predictions["predicted"], labels=CLASS_IDS)
    matrix = pd.DataFrame(counts, index=CLASSES, columns=CLASSES).rename_axis("actual_sport")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(RESULTS_DIR / "validation_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    display = ConfusionMatrixDisplay(confusion_matrix=counts, display_labels=CLASSES)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set(xlabel="Predicted sport", ylabel="Actual sport",
           title=f"Softmax validation confusion matrix\nFive student-grouped folds, {len(predictions)} responses")
    fig.savefig(RESULTS_DIR / "validation_confusion_matrix.png", dpi=240)
    plt.close(fig)
    print("\nCombined validation counts (rows: actual, columns: predicted)")
    print(matrix.to_string())


if __name__ == "__main__":
    main()
