"""Report softmax performance on the validation folds and local test set."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from data_exploration.dataexploration import load_dataset, split_by_student
from data_exploration.preprocessing import SurveyPreprocessor
from models.softmax.compare_softmax_features import with_text_columns
from models.softmax.softmax_regression import (
    CLASSES, MAX_EPOCHS, SoftmaxRegression, _encode_labels, prepare_fold,
)


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
        print(f"Fold {fold_number}: {len(y_val)} validation rows, {model.n_epochs_} epochs")

    combined = pd.concat(predictions_by_fold).sort_index()
    development_rows = data.index[~test_mask].sort_values()
    if not combined.index.is_unique or not combined.index.equals(development_rows):
        raise ValueError("Validation predictions must cover each development row exactly once.")
    return combined


def summarize_validation_metrics(predictions):
    """Compute the two proposal metrics per fold and summarize across folds."""
    rows = []
    for fold, responses in predictions.groupby("fold"):
        actual, predicted = responses["actual"], responses["predicted"]
        rows.append({
            "fold": fold,
            "validation_rows": len(responses),
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, labels=CLASS_IDS, average="macro"),
        })
    by_fold = pd.DataFrame(rows).set_index("fold")
    summary = by_fold[["accuracy", "macro_f1"]].agg(["mean", "std"]).rename_axis("statistic")
    return by_fold, summary


def evaluate_heldout_test(data):
    """Refit on the development set and return test metrics and confusion counts."""
    test_mask, _ = split_by_student(data["id"])
    development = data.loc[~test_mask]
    test = data.loc[test_mask]

    preprocessor = SurveyPreprocessor().fit(development)
    X_development = preprocessor.transform(development, model="softmax")
    columns = with_text_columns(X_development.columns)
    y_development = _encode_labels(development["label"])
    model = SoftmaxRegression(**BEST_SETTINGS).fit(X_development.loc[:, columns], y_development)

    X_test = preprocessor.transform(test, model="softmax")
    predictions = model.predict(X_test.loc[:, columns])
    y_test = _encode_labels(test["label"])
    metrics = {
        "development_rows": len(development),
        "test_rows": len(test),
        "correct": int((y_test.to_numpy() == predictions).sum()),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, labels=CLASS_IDS, average="macro"),
    }
    return metrics, confusion_matrix(y_test, predictions, labels=CLASS_IDS)


def save_confusion_matrix(counts, filename, title):
    """Save a labeled count table and a plot of the same matrix."""
    output = RESULTS_DIR / filename
    matrix = pd.DataFrame(counts, index=CLASSES, columns=CLASSES).rename_axis("actual_sport")
    matrix.to_csv(output.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    display = ConfusionMatrixDisplay(confusion_matrix=counts, display_labels=CLASSES)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set(xlabel="Predicted sport", ylabel="Actual sport", title=title)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return matrix


def main():
    print(f"Selected softmax settings: {BEST_SETTINGS}; epoch cap: {MAX_EPOCHS}")
    data = load_dataset()
    predictions = collect_validation_predictions(data)
    by_fold, summary = summarize_validation_metrics(predictions)
    counts = confusion_matrix(predictions["actual"], predictions["predicted"], labels=CLASS_IDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(RESULTS_DIR / "validation_metrics_by_fold.csv")
    summary.to_csv(RESULTS_DIR / "validation_metrics_summary.csv")
    matrix = save_confusion_matrix(
        counts, "validation_confusion_matrix",
        f"Softmax validation confusion matrix\nFive student-grouped folds, {len(predictions)} responses",
    )
    print("\nValidation metrics by fold")
    print(by_fold.round(6).to_string())
    print("\nMean and standard deviation across the five folds")
    print(summary.round(6).to_string())
    print("\nCombined validation counts (rows: actual, columns: predicted)")
    print(matrix.to_string())

    test_metrics, test_counts = evaluate_heldout_test(data)
    test_output = RESULTS_DIR / "test_metrics.csv"
    pd.DataFrame([test_metrics]).to_csv(test_output, index=False)
    test_matrix = save_confusion_matrix(
        test_counts, "test_confusion_matrix",
        f"Softmax test confusion matrix\n20% student holdout, {test_metrics['test_rows']} responses",
    )
    print("\nLocal held-out test results")
    print(f"Refit on {test_metrics['development_rows']} development responses.")
    print(f"Accuracy: {test_metrics['accuracy']:.2%} "
          f"({test_metrics['correct']}/{test_metrics['test_rows']} correct)")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Saved test metrics to {test_output}")
    print("\nTest counts (rows: actual, columns: predicted)")
    print(test_matrix.to_string())


if __name__ == "__main__":
    main()
