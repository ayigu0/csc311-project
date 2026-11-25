"""Comparable three-model reporting on the team's exact student-grouped folds.

Run: python models/random_forest/evaluate_shared_folds.py
Use the shared 20% holdout and five development folds. Keep all selected model
settings and model-specific preprocessing unchanged; fit preprocessing in each
training fold. Existing teammate outputs and historical RF results are preserved.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.dataexploration import DATA_FILE, load_dataset, split_by_student
from models.random_forest.reporting_utils import sha256
from models.random_forest.submit.random_forest import PARAMETERS_FILE, RandomForestModel
from models.softmax import evaluate_softmax
from models.naive_bayes import evaluate_naive_bayes

CLASSES = ["Baseball", "Basketball", "Soccer"]


def validate_predictions(name, predictions, data, test, folds):
    if predictions.index.has_duplicates or not predictions.index.equals(data.index[~test].sort_values()):
        raise ValueError(f"{name}: each development row must be predicted once, with no test rows.")
    if predictions.actual.tolist() != data.loc[predictions.index, "label"].tolist():
        raise ValueError(f"{name}: labels are not aligned with original rows.")
    if not predictions.predicted.isin(CLASSES).all():
        raise ValueError(f"{name}: invalid predicted class.")
    for fold, (_, validation) in enumerate(folds, 1):
        if predictions.index[predictions.fold.eq(fold)].tolist() != data.index[validation].tolist():
            raise ValueError(f"{name}: validation membership differs in fold {fold}.")


def main():
    if (evaluate_softmax.split_by_student is not split_by_student
            or evaluate_naive_bayes.split_by_student is not split_by_student):
        raise ValueError("Models do not share the same split implementation.")
    # Shared identity/label table; RF keeps its selected raw-reading behavior.
    data = load_dataset()
    rf_raw = pd.read_csv(DATA_FILE)
    if data.id.tolist() != rf_raw.id.tolist() or data.label.tolist() != rf_raw.label.tolist():
        raise ValueError("RF raw rows differ from team data.")
    test, folds = split_by_student(data.id)
    settings = json.loads(PARAMETERS_FILE.read_text())
    nb_summary = PROJECT_ROOT / "results/naive_bayes/tuning_summary.csv"
    teammate_files = [p for folder in ["softmax", "naive_bayes"]
                      for p in (PROJECT_ROOT / "results" / folder).rglob("*") if p.is_file()]
    original_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in teammate_files}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = PROJECT_ROOT / "results/random_forest/runs" / f"shared_cv_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    assignments = data[["id"]].copy()
    assignments["role"] = np.where(test, "test_reserved", "development")
    assignments["validation_fold"] = 0
    predictions, rf_frames, rf_feature_counts = {}, [], {}

    for fold, (training, validation) in enumerate(folds, 1):
        train_ids, val_ids, test_ids = (set(data.loc[mask, "id"]) for mask in [training, validation, test])
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"Student leakage in fold {fold}.")
        if not (training | validation | test).all():
            raise ValueError("Missing partition assignments.")
        assignments.loc[validation, "validation_fold"] = fold
        model = RandomForestModel().fit(rf_raw.loc[training])
        predicted = model.predict(rf_raw.loc[validation].drop(columns=["id", "label"]))
        rf_frames.append(pd.DataFrame({"fold": fold, "actual": data.loc[validation, "label"],
                                       "predicted": predicted}, index=data.index[validation]))
        rf_feature_counts[fold] = len(model.preprocessing_state_["feature_columns"])
        (output / f"rf_fold_{fold}_preprocessing.json").write_text(json.dumps(model.preprocessing_state_, indent=2) + "\n")
        print(f"RF fold {fold}: {training.sum()} train / {validation.sum()} validation rows", flush=True)
    predictions["random_forest_gini"] = pd.concat(rf_frames).sort_index()

    print("Replaying Softmax's existing evaluation without overwriting its files...", flush=True)
    softmax = evaluate_softmax.collect_validation_predictions(data)
    class_names = dict(enumerate(evaluate_softmax.CLASSES))
    softmax = softmax.assign(actual=softmax.actual.map(class_names), predicted=softmax.predicted.map(class_names))
    predictions["softmax"] = softmax

    print("Replaying Naive Bayes' existing selected evaluation...", flush=True)
    nb, _ = evaluate_naive_bayes.evaluate(data, nb_summary)
    predictions["naive_bayes"] = nb
    previous_nb = pd.read_csv(PROJECT_ROOT / "results/naive_bayes/validation_predictions.csv", index_col="row_index")
    pd.testing.assert_frame_equal(nb, previous_nb, check_names=False)

    rows, summaries = [], []
    for name, prediction in predictions.items():
        validate_predictions(name, prediction, data, test, folds)
        prediction.to_csv(output / f"{name}_validation_predictions.csv", index_label="source_row_index")
        for fold, (training, validation) in enumerate(folds, 1):
            part = prediction[prediction.fold.eq(fold)]
            rows.append({
                "model": name, "fold": fold,
                "training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
                "training_students": data.loc[training, "id"].nunique(),
                "validation_students": data.loc[validation, "id"].nunique(),
                "accuracy": accuracy_score(part.actual, part.predicted),
                "macro_f1": f1_score(part.actual, part.predicted, labels=CLASSES, average="macro", zero_division=0),
            })
        scores = pd.DataFrame([row for row in rows if row["model"] == name])
        summaries.append({
            "model": name, "folds": len(folds), "development_rows": int((~test).sum()),
            "accuracy_mean": scores.accuracy.mean(), "accuracy_std": scores.accuracy.std(ddof=1),
            "macro_f1_mean": scores.macro_f1.mean(), "macro_f1_std": scores.macro_f1.std(ddof=1),
            "pooled_oof_accuracy": accuracy_score(prediction.actual, prediction.predicted),
            "pooled_oof_macro_f1": f1_score(prediction.actual, prediction.predicted, labels=CLASSES, average="macro"),
        })
        pd.DataFrame(confusion_matrix(prediction.actual, prediction.predicted, labels=CLASSES),
                     index=pd.Index(CLASSES, name="actual"), columns=pd.Index(CLASSES, name="predicted")).to_csv(output / f"{name}_confusion_matrix.csv")
        pd.DataFrame(classification_report(prediction.actual, prediction.predicted, labels=CLASSES, output_dict=True, zero_division=0)).T.to_csv(output / f"{name}_classification_report.csv")
    by_fold, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    old_softmax = pd.read_csv(PROJECT_ROOT / "results/softmax/validation_metrics_by_fold.csv")
    current_softmax = by_fold[by_fold.model.eq("softmax")]
    np.testing.assert_allclose(current_softmax[["accuracy", "macro_f1"]], old_softmax[["accuracy", "macro_f1"]])
    assignments.to_csv(output / "shared_fold_assignments.csv", index_label="source_row_index")
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    for path, checksum in original_hashes.items():
        if sha256(PROJECT_ROOT / path) != checksum:
            raise ValueError(f"Teammate artifact changed: {path}")
    provenance_paths = ["data_exploration/dataexploration.py", "data_exploration/preprocessing.py",
                        "models/softmax/evaluate_softmax.py", "models/softmax/softmax_regression.py",
                        "models/naive_bayes/evaluate_naive_bayes.py", "models/naive_bayes/naive_bayes.py"]
    config = {
        "protocol": "shared 20% student holdout, five development folds; same exact rows per fold for all three models",
        "test_fraction": 0.2, "split_seed": 42, "fold_seed": 43, "n_folds": len(folds),
        "development_rows": int((~test).sum()), "reserved_test_rows": int(test.sum()),
        "development_students": data.loc[~test, "id"].nunique(), "reserved_test_students": data.loc[test, "id"].nunique(),
        "reporting_metric": "unweighted mean and sample standard deviation over five validation folds; pooled OOF is separate",
        "rf_model_options": settings["model_options"], "rf_parameters_sha256": sha256(PARAMETERS_FILE),
        "rf_package_manifest_sha256": sha256(PARAMETERS_FILE.with_name("package_manifest.json")),
        "rf_preprocessing": "selected RF v1, freshly fitted per training fold; same input parsing as selected package experiments",
        "rf_features_by_fold": rf_feature_counts, "softmax_settings": evaluate_softmax.BEST_SETTINGS,
        "naive_bayes_selected_settings": pd.read_csv(nb_summary).iloc[0].to_dict(),
        "teammate_outputs_exactly_reproduced": True, "teammate_artifacts_unchanged": original_hashes,
        "source_sha256": {path: sha256(PROJECT_ROOT / path) for path in provenance_paths},
        "raw_data_sha256": sha256(DATA_FILE), "test_evaluated": False, "new_tuning": False,
        "limitations": "Different model-specific preprocessing and historical tuning procedures remain. These are matched development CV results, not nested CV or untouched test results. RF previously used local-holdout data in other experiments.",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output / "comparison_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Matched report results: {output}")
    return output


if __name__ == "__main__":
    main()
