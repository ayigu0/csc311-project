"""Select RF preprocessing on the old validation split, then run grouped 5-fold.

Run: python models/random_forest/compare_data_and_cross_validate.py
Keep the original 30% student holdout. Refit preprocessing separately inside
each development fold. CV estimates stability of previously selected settings;
it is not nested CV or a new, independent test estimate.
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

from models.random_forest.data_preprocessing import (
    DATA_FILE, FEATURE_FILE, METADATA_FILE, LABELS, base_features, clean_answers,
    fit_preprocessor, transform_features, load_preprocessed_dataset, sha256,
)
from models.random_forest.main_reference.preprocessing import (
    SurveyPreprocessor, normalize_answers, numeric_answers,
)
from models.random_forest.random_forest import build_pipeline

HERE = Path(__file__).resolve().parent
N_FOLDS, FOLD_SEED = 5, 43


def prepare(name, train, validation):
    # Labels and student IDs are excluded even when fitting preprocessing.
    train, validation = (frame.drop(columns=["id", "label"]) for frame in [train, validation])
    if name == "rf_v1":
        state = fit_preprocessor(train)
        return transform_features(train, state), transform_features(validation, state), state
    preprocessor = SurveyPreprocessor().fit(train)
    state = {
        "imputation": preprocessor.imputation_.to_dict(),
        "options": preprocessor.options_,
        "vocabularies": {field: list(vectorizer.get_feature_names_out())
                         for field, (vectorizer, _) in preprocessor.text_encoders_.items()},
    }
    return preprocessor.transform(train, model="random_forest"), preprocessor.transform(validation, model="random_forest"), state


def check_features(X_train, X_val, train, val):
    if (not X_train.index.equals(train.index) or not X_val.index.equals(val.index)
            or not X_train.columns.equals(X_val.columns)
            or set(X_train.columns) & {"id", "label", "split", "source_row_index"}
            or not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_val.to_numpy()).all()):
        raise ValueError("Feature validity or alignment check failed.")


def main():
    saved, metadata = load_preprocessed_dataset()
    config_path = HERE / "tuned_params.json"
    settings = json.loads(config_path.read_text())
    if settings["data_sha256"] != sha256(FEATURE_FILE) or settings["preprocessing_metadata_sha256"] != sha256(METADATA_FILE):
        raise ValueError("Tuned settings do not match the current RF data version.")
    reference = json.loads((HERE / "main_reference/source.json").read_text())
    if reference["raw_csv_sha256"] != sha256(DATA_FILE):
        raise ValueError("Main and RF raw datasets differ; align observations before comparing.")
    for name, checksum in reference["source_sha256"].items():
        if sha256(HERE / "main_reference" / Path(name).name) != checksum:
            raise ValueError("Pinned main source was modified.")
    inputs = {"rf_v1": pd.read_csv(DATA_FILE), "main": pd.read_csv(DATA_FILE, keep_default_na=False)}
    raw = inputs["rf_v1"]
    if raw.index.tolist() != saved.index.tolist() or raw.id.tolist() != saved.id.tolist():
        raise ValueError("Raw rows no longer align with the saved split.")
    if saved.groupby("id")["split"].nunique().ne(1).any():
        raise ValueError("Student overlap in saved split.")
    train_mask, val_mask, test_mask = (saved["split"].eq(name) for name in ["train", "validation", "test"])
    options = settings["model_options"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = PROJECT_ROOT / "results" / f"random_forest_5fold_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    comparison, comparison_predictions = [], []
    for name, frame in inputs.items():
        train, val = frame.loc[train_mask], frame.loc[val_mask]
        X_train, X_val, _ = prepare(name, train, val)
        check_features(X_train, X_val, train, val)
        model = build_pipeline(**options).fit(X_train, train["label"])
        prediction = model.predict(X_val)
        numeric = base_features(clean_answers(train)) if name == "rf_v1" else numeric_answers(normalize_answers(train))
        price_col = "ticket_price_cad" if name == "rf_v1" else "ticket_price"
        comparison.append({
            "preprocessing": name, "features": X_train.shape[1],
            "validation_accuracy": accuracy_score(val["label"], prediction),
            "validation_macro_f1": f1_score(val["label"], prediction, labels=LABELS, average="macro"),
            "training_prices_missing_or_unparsed": int(numeric[price_col].isna().sum()),
            "finite_features": True, "row_and_column_alignment": True,
        })
        comparison_predictions.append(pd.DataFrame({"preprocessing": name, "source_row_index": val.index,
                                                     "actual": val["label"].to_numpy(), "predicted": prediction}))
    selected = max(comparison, key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"], row["preprocessing"] == "rf_v1"))["preprocessing"]
    pd.DataFrame(comparison).to_csv(output / "preprocessing_comparison.csv", index=False)
    pd.concat(comparison_predictions, ignore_index=True).to_csv(output / "preprocessing_validation_predictions.csv", index=False)
    print(f"Selected preprocessing before CV: {selected}", flush=True)

    frame = inputs[selected]
    dev_ids = sorted(frame.loc[~test_mask, "id"].unique())
    heldout_ids = set(frame.loc[test_mask, "id"])
    validation_groups = np.array_split(np.random.default_rng(FOLD_SEED).permutation(dev_ids), N_FOLDS)
    assignments = saved[["id"]].copy()
    assignments["role"] = np.where(test_mask, "test_reserved", "development")
    assignments["validation_fold"] = 0
    metrics, predictions = [], []
    for fold, ids in enumerate(validation_groups, 1):
        val_mask = frame["id"].isin(ids)
        train_mask = ~test_mask & ~val_mask
        train, val = frame.loc[train_mask], frame.loc[val_mask]
        if set(train.id) & set(val.id) or (set(train.id) | set(val.id)) & heldout_ids:
            raise ValueError("Student leakage in cross-validation folds.")
        assignments.loc[val_mask, "validation_fold"] = fold
        X_train, X_val, state = prepare(selected, train, val)
        check_features(X_train, X_val, train, val)
        (output / f"fold_{fold}_preprocessing.json").write_text(json.dumps(state, indent=2) + "\n")
        for criterion in ["gini", "entropy"]:
            model = build_pipeline(**{**options, "criterion": criterion}).fit(X_train, train["label"])
            train_prediction, prediction = model.predict(X_train), model.predict(X_val)
            metrics.append({
                "criterion": criterion, "fold": fold, "features": X_train.shape[1],
                "training_rows": len(train), "validation_rows": len(val),
                "training_students": train.id.nunique(), "validation_students": val.id.nunique(),
                "train_accuracy": accuracy_score(train["label"], train_prediction),
                "accuracy": accuracy_score(val["label"], prediction),
                "macro_f1": f1_score(val["label"], prediction, labels=LABELS, average="macro"),
            })
            predictions.append(pd.DataFrame({"criterion": criterion, "fold": fold,
                                             "source_row_index": val.index, "actual": val["label"].to_numpy(),
                                             "predicted": prediction}))
        print(f"Fold {fold}: {len(train)} training / {len(val)} validation rows; {X_train.shape[1]} features", flush=True)
    by_fold = pd.DataFrame(metrics)
    oof = pd.concat(predictions, ignore_index=True)
    summaries = []
    for criterion in ["gini", "entropy"]:
        part, scores = oof[oof.criterion.eq(criterion)], by_fold[by_fold.criterion.eq(criterion)]
        if part.source_row_index.duplicated().any() or set(part.source_row_index) != set(frame.index[~test_mask]):
            raise ValueError("Each development row must receive exactly one out-of-fold prediction.")
        summaries.append({
            "criterion": criterion, "accuracy_mean": scores.accuracy.mean(), "accuracy_std": scores.accuracy.std(ddof=1),
            "macro_f1_mean": scores.macro_f1.mean(), "macro_f1_std": scores.macro_f1.std(ddof=1),
            "pooled_oof_accuracy": accuracy_score(part.actual, part.predicted),
            "pooled_oof_macro_f1": f1_score(part.actual, part.predicted, labels=LABELS, average="macro"),
        })
        pd.DataFrame(confusion_matrix(part.actual, part.predicted, labels=LABELS),
                     index=pd.Index(LABELS, name="actual"), columns=pd.Index(LABELS, name="predicted")).to_csv(output / f"{criterion}_oof_confusion_matrix.csv")
        pd.DataFrame(classification_report(part.actual, part.predicted, labels=LABELS, output_dict=True, zero_division=0)).T.to_csv(output / f"{criterion}_oof_classification_report.csv")
    summary = pd.DataFrame(summaries)
    assignments.to_csv(output / "fold_assignments.csv", index_label="source_row_index")
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    record = {
        "selected_preprocessing": selected, "main_reference": reference,
        "raw_data_sha256": sha256(DATA_FILE), "rf_preprocessing_code_sha256": sha256(HERE / "data_preprocessing.py"),
        "tuned_config_sha256": sha256(config_path), "model_options": options,
        "criteria_evaluated": ["gini", "entropy"], "primary_criterion_selected_before_cv": options["criterion"],
        "note_on_entropy": "Same frozen structural parameters as Gini, with only criterion changed; no CV tuning.",
        "preprocessing_selection": "old 435/174 train/validation split, fixed tuned Gini, validation macro-F1 then accuracy",
        "fold_seed": FOLD_SEED, "n_folds": N_FOLDS, "student_grouped": True,
        "development_rows": int((~test_mask).sum()), "development_students": len(dev_ids),
        "reserved_test_rows": int(test_mask.sum()), "reserved_test_students": len(heldout_ids),
        "test_evaluated": False, "preprocessing_refit_each_fold": True,
        "limitation": "Previously selected hyperparameters and preprocessing; reused development data. Not nested CV or a fresh independent test estimate; local test was evaluated earlier, not in this run.",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output / "cv_config.json").write_text(json.dumps(record, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results: {output}")
    return output


if __name__ == "__main__":
    main()
