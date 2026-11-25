"""Refit the selected Naive Bayes model and export the standalone prediction state.

"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_exploration.dataexploration import load_dataset, rename_features
from data_exploration.preprocessing import SurveyPreprocessor
from models.naive_bayes.evaluate_naive_bayes import load_selected_settings
from models.naive_bayes.naive_bayes import CLASSES, MixedNaiveBayes
from models.naive_bayes.tune_naive_bayes import RESULTS_DIR
from pred import PARAMETERS_FILE, _decision_scores, _prepare_features


def train_submission():
    """Train on every labeled row with the settings selected in cross-validation."""
    data = load_dataset()
    preprocessing_settings, model_settings = load_selected_settings(RESULTS_DIR / "tuning_summary.csv")
    preprocessor = SurveyPreprocessor(**preprocessing_settings).fit(data)
    features = preprocessor.transform(data, model="naive_bayes")
    model = MixedNaiveBayes(**model_settings).fit(features, data["label"])

    preprocessing = {
        "column_names": {
            raw: short for raw, short in zip(data.columns, rename_features(data).columns) if raw != short
        },
        "imputation": preprocessor.imputation_.to_dict(),
        "options": preprocessor.options_,
        "vocabularies": {
            field: vectorizer.get_feature_names_out().tolist()
            for field, (vectorizer, _) in preprocessor.text_encoders_.items()
        },
    }
    parameters = {
        "classes": list(CLASSES),
        "binary_columns": model.binary_columns_,
        "binary_weights": model.binary_weights_.tolist(),
        "log_prior": np.asarray(model.log_prior_).tolist(),
        "means": np.asarray(model.means_).tolist(),
        "variances": np.asarray(model.variances_).tolist(),
        "categorical_log_probs": np.asarray(model.categorical_log_probs_).tolist(),
        "binary_log_present": np.asarray(model.binary_log_present_).tolist(),
        "binary_log_absent": np.asarray(model.binary_log_absent_).tolist(),
    }
    bundle = {
        "format_version": 1,
        "training_rows": len(data),
        "preprocessing_settings": preprocessing_settings,
        "model_settings": model_settings,
        "preprocessing": preprocessing,
        "model": parameters,
    }
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"

    # Test the same serialized state and code that the submitted predictor will use.
    restored = json.loads(encoded)
    submission_features = _prepare_features(data.drop(columns=["id", "label"]), restored["preprocessing"])
    pd.testing.assert_frame_equal(features, submission_features, check_exact=True)
    np.testing.assert_array_equal(
        model.decision_function(features),
        _decision_scores(submission_features, restored["model"]),
    )

    with TemporaryDirectory(prefix=".naive-bayes-", dir=PARAMETERS_FILE.parent) as temporary:
        staged = Path(temporary) / PARAMETERS_FILE.name
        staged.write_text(encoded, encoding="utf-8")
        staged.replace(PARAMETERS_FILE)
    print(f"Fitted Naive Bayes on {len(data)} labeled responses from {data['id'].nunique()} students.")
    print(f"Saved {PARAMETERS_FILE.name} ({PARAMETERS_FILE.stat().st_size:,} bytes).")


if __name__ == "__main__":
    train_submission()
