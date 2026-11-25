"""Summarize out-of-fold errors and the text-token ablation."""

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


def evaluate(data, summary_path: Path):
    best = pd.read_csv(summary_path).iloc[0]
    preprocessor_settings = {
        "min_text_frequency": int(best["min_text_frequency"]),
        "max_terms_per_field": int(best["max_terms_per_field"]),
    }
    model_settings = {
        "alpha": float(best["alpha"]),
        "text_weight": float(best["text_weight"]),
        "var_smoothing": float(best["var_smoothing"]),
    }
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


def save_confusion_plot(counts, prediction_count):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Install matplotlib from requirements.txt to also generate the PNG.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay(counts, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set(xlabel="Predicted sport", ylabel="Actual sport",
           title=f"Naive Bayes validation confusion matrix (n={prediction_count})")
    fig.savefig(RESULTS_DIR / "validation_confusion_matrix.png", dpi=200)
    plt.close(fig)


def main():
    summary_path = RESULTS_DIR / "tuning_summary.csv"
    if not summary_path.is_file():
        raise SystemExit("Run python -m models.naive_bayes.tune_naive_bayes first.")

    predictions, ablation_scores = evaluate(load_dataset(), summary_path)
    predictions.to_csv(RESULTS_DIR / "validation_predictions.csv", index_label="row_index")
    ablation_scores.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)

    counts = confusion_matrix(predictions["actual"], predictions["predicted"], labels=CLASSES)
    matrix = pd.DataFrame(counts, index=CLASSES, columns=CLASSES).rename_axis("actual_sport")
    matrix.to_csv(RESULTS_DIR / "validation_confusion_matrix.csv")

    print(matrix)
    metric_columns = ["validation_accuracy", "validation_macro_f1"]
    print(ablation_scores.groupby("variant")[metric_columns].mean().round(6))
    save_confusion_plot(counts, len(predictions))


if __name__ == "__main__":
    main()
