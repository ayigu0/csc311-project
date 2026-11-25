"""Evaluate Naive Bayes on the shared validation folds or held-out students."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from data_exploration.dataexploration import load_dataset, split_by_student
from models.naive_bayes.naive_bayes import (
    CLASSES,
    MixedNaiveBayes,
    prepare_fold,
    score_predictions,
)
from models.naive_bayes.tune_naive_bayes import RESULTS_DIR


def load_selected_settings(summary_path: Path):
    """Read the winner of the development-fold tuning run."""
    best = pd.read_csv(summary_path, nrows=1).iloc[0]
    preprocessor_settings = {
        "min_text_frequency": int(best["min_text_frequency"]),
        "max_terms_per_field": int(best["max_terms_per_field"]),
    }
    model_settings = {
        "alpha": float(best["alpha"]),
        "text_weight": float(best["text_weight"]),
        "var_smoothing": float(best["var_smoothing"]),
    }
    return preprocessor_settings, model_settings


def evaluate(data, summary_path: Path):
    preprocessor_settings, model_settings = load_selected_settings(summary_path)
    test_mask, folds = split_by_student(data["id"])
    prediction_frames = []
    scores = []

    for fold, (train_mask, validation_mask) in enumerate(folds, start=1):
        train, validation = data.loc[train_mask], data.loc[validation_mask]
        X_train, X_validation = prepare_fold(train, validation, **preprocessor_settings)
        actual = validation["label"]

        for variant, text_weight in (
            ("full_model", model_settings["text_weight"]),
            ("without_text_tokens", 0.0),
        ):
            settings = {**model_settings, "text_weight": text_weight}
            predicted = MixedNaiveBayes(**settings).fit(
                X_train, train["label"],
            ).predict(X_validation)
            scores.append({
                "variant": variant, "fold": fold,
                **score_predictions(actual, predicted),
            })
            if variant == "full_model":
                prediction_frames.append(pd.DataFrame(
                    {"fold": fold, "actual": actual, "predicted": predicted},
                    index=validation.index,
                ))

    predictions = pd.concat(prediction_frames, verify_integrity=True).sort_index()
    if not predictions.index.equals(data.index[~test_mask].sort_values()):
        raise ValueError("Expected exactly one prediction per development row.")
    return predictions, pd.DataFrame(scores)


def evaluate_heldout_test(data, summary_path: Path):
    """Refit on the full development set and score the held-out students."""
    preprocessor_settings, model_settings = load_selected_settings(summary_path)
    test_mask, _ = split_by_student(data["id"])
    development, test = data.loc[~test_mask], data.loc[test_mask]
    X_development, X_test = prepare_fold(development, test, **preprocessor_settings)
    model = MixedNaiveBayes(**model_settings).fit(X_development, development["label"])
    actual = test["label"]
    predicted = model.predict(X_test)
    scores = score_predictions(actual, predicted)
    metrics = {
        "development_rows": len(development),
        "test_rows": len(test),
        "correct": int((actual.to_numpy() == predicted).sum()),
        "accuracy": scores["validation_accuracy"],
        "macro_f1": scores["validation_macro_f1"],
    }
    return metrics, confusion_matrix(actual, predicted, labels=CLASSES)


def save_confusion_matrix(counts, partition):
    """Save the confusion counts and, when available, a labeled plot."""
    output = RESULTS_DIR / f"{partition}_confusion_matrix"
    matrix = pd.DataFrame(counts, index=CLASSES, columns=CLASSES).rename_axis("actual_sport")
    matrix.to_csv(output.with_suffix(".csv"))
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Install matplotlib from requirements.txt to also generate the PNG.")
        return matrix
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay(counts, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set(xlabel="Predicted sport", ylabel="Actual sport",
           title=f"Naive Bayes {partition} confusion matrix (n={counts.sum()})")
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return matrix


def main(*, held_out_test=False):
    summary_path = RESULTS_DIR / "tuning_summary.csv"
    if not summary_path.is_file():
        raise SystemExit("Run python -m models.naive_bayes.tune_naive_bayes first.")

    data = load_dataset()
    if held_out_test:
        metrics, counts = evaluate_heldout_test(data, summary_path)
        test_output = RESULTS_DIR / "test_metrics.csv"
        pd.DataFrame([metrics]).to_csv(test_output, index=False)
        matrix = save_confusion_matrix(counts, "test")
        print(f"Refit on {metrics['development_rows']} development responses.")
        print(f"Held-out accuracy: {metrics['accuracy']:.2%} "
              f"({metrics['correct']}/{metrics['test_rows']} correct)")
        print(f"Held-out macro F1: {metrics['macro_f1']:.4f}")
        print(f"Saved test metrics to {test_output}")
        print(matrix)
        return

    predictions, ablation_scores = evaluate(data, summary_path)
    predictions.to_csv(RESULTS_DIR / "validation_predictions.csv", index_label="row_index")
    ablation_scores.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)

    counts = confusion_matrix(predictions["actual"], predictions["predicted"], labels=CLASSES)
    matrix = save_confusion_matrix(counts, "validation")

    print(matrix)
    metric_columns = ["validation_accuracy", "validation_macro_f1"]
    print(ablation_scores.groupby("variant")[metric_columns].mean().round(6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-test", action="store_true",
                        help="Refit the selected model and evaluate only the held-out test set.")
    main(held_out_test=parser.parse_args().held_out_test)
