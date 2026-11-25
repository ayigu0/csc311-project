"""Selected RF model; this module imports only files inside submit and libraries.

Training rows are supplied by the caller. This module does not choose a split,
tune parameters, or run experiments. sklearn is required for model training
and prediction; this is not the course's restricted-library pred.py interface.
"""

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

if __package__:
    from .data_preprocessing import LABELS, fit_preprocessor, transform_features
else:
    from data_preprocessing import LABELS, fit_preprocessor, transform_features

PARAMETERS_FILE = Path(__file__).with_name("tuned_params.json")


class RandomForestModel:
    """Fit the selected Gini forest and its preprocessing on supplied raw rows."""

    def __init__(self, parameters_file=PARAMETERS_FILE):
        config = json.loads(Path(parameters_file).read_text(encoding="utf-8"))
        self.model_options = config["model_options"].copy()

    def fit(self, training_frame):
        if training_frame.empty:
            raise ValueError("Training rows must not be empty.")
        if "label" not in training_frame or not training_frame["label"].isin(LABELS).all():
            raise ValueError("Training rows must have valid sport labels.")
        answers = training_frame.drop(columns=["id", "label"], errors="ignore")
        self.preprocessing_state_ = fit_preprocessor(answers)
        features = transform_features(answers, self.preprocessing_state_)
        options = self.model_options.copy()
        options["random_state"] = options.pop("seed")
        self.forest_ = RandomForestClassifier(**options, n_jobs=-1)
        self.forest_.fit(features, training_frame["label"])
        return self

    def predict(self, raw_frame):
        """Predict in input order; labels and student IDs are not required."""
        if not hasattr(self, "forest_"):
            raise ValueError("Fit the model before predicting.")
        features = transform_features(raw_frame, self.preprocessing_state_)
        if features.empty:
            return []
        return self.forest_.predict(features).tolist()
