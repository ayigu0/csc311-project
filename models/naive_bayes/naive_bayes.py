"""Mixed Naive Bayes for the sport survey."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_exploration.dataexploration import (
    EASE_COL,
    LIKERT_COLS,
    load_dataset,
    split_by_student,
)
from data_exploration.preprocessing import PRICE, SurveyPreprocessor


CLASSES = ("Baseball", "Basketball", "Soccer")
CONTINUOUS = (EASE_COL, PRICE)
LIKERT_VALUES = np.arange(1, 6)


class MixedNaiveBayes:
    """Combine Gaussian, categorical, and Bernoulli log-likelihoods."""

    def __init__(self, alpha=1.0, text_weight=1.0, var_smoothing=0.01):
        self.alpha = alpha
        self.text_weight = text_weight
        self.var_smoothing = var_smoothing

    def _feature_blocks(self, features):
        continuous = features.loc[:, list(CONTINUOUS)].to_numpy(dtype=float, copy=True)
        continuous[:, 1] = np.log1p(continuous[:, 1])
        categorical = features.loc[:, LIKERT_COLS].to_numpy(dtype=int) - 1
        binary = features.loc[:, self.binary_columns_].to_numpy(dtype=float)
        return continuous, categorical, binary

    def fit(self, features, labels):
        self.binary_columns_ = [
            column for column in features
            if column not in CONTINUOUS and column not in LIKERT_COLS
        ]
        self.binary_weights_ = np.array([
            self.text_weight if "__token__" in column else 1.0
            for column in self.binary_columns_
        ])
        continuous, categorical, binary = self._feature_blocks(features)
        labels = np.asarray(labels)
        if labels.shape != (len(features),) or not np.isin(labels, CLASSES).all():
            raise ValueError("Expected one valid sport label for each feature row.")
        # The floor is specific to each feature and uses training rows only.
        variance_floor = self.var_smoothing * np.maximum(continuous.var(axis=0), 1e-12)

        self.log_prior_ = []
        self.means_ = []
        self.variances_ = []
        self.categorical_log_probs_ = []
        self.binary_log_present_ = []
        self.binary_log_absent_ = []

        for sport in CLASSES:
            rows = labels == sport
            count = rows.sum()
            if count == 0:
                raise ValueError(f"Training split is missing the class {sport}.")
            self.log_prior_.append(np.log(count / len(labels)))
            self.means_.append(continuous[rows].mean(axis=0))
            self.variances_.append(np.maximum(continuous[rows].var(axis=0), variance_floor))

            # Each Likert question has its own five-category distribution.
            likert_counts = np.stack([
                np.bincount(categorical[rows, question], minlength=len(LIKERT_VALUES))
                for question in range(len(LIKERT_COLS))
            ])
            probabilities = (likert_counts + self.alpha) / (count + 5 * self.alpha)
            self.categorical_log_probs_.append(np.log(probabilities))

            present = (binary[rows].sum(axis=0) + self.alpha) / (count + 2 * self.alpha)
            self.binary_log_present_.append(np.log(present))
            self.binary_log_absent_.append(np.log1p(-present))
        return self

    def decision_function(self, features):
        """Return one score per row and sport, in CLASSES order."""
        continuous, categorical, binary = self._feature_blocks(features)
        scores = np.empty((len(features), len(CLASSES)))
        for index, log_prior in enumerate(self.log_prior_):
            variance = self.variances_[index]
            squared_distance = (continuous - self.means_[index]) ** 2
            gaussian = -0.5 * (np.log(2 * np.pi * variance) + squared_distance / variance)
            likert = self.categorical_log_probs_[index][
                np.arange(len(LIKERT_COLS)), categorical
            ]
            bernoulli = (
                binary * self.binary_log_present_[index]
                + (1 - binary) * self.binary_log_absent_[index]
            )
            scores[:, index] = (
                log_prior + gaussian.sum(axis=1) + likert.sum(axis=1)
                + (bernoulli * self.binary_weights_).sum(axis=1)
            )
        return scores

    def predict(self, features):
        return np.asarray(CLASSES)[self.decision_function(features).argmax(axis=1)]


def prepare_fold(train, validation, min_text_frequency=2, max_terms_per_field=200):
    """Build matching feature columns using statistics from the training rows."""
    preprocessor = SurveyPreprocessor(
        min_text_frequency=min_text_frequency,
        max_terms_per_field=max_terms_per_field,
    ).fit(train)
    return (
        preprocessor.transform(train, model="naive_bayes"),
        preprocessor.transform(validation, model="naive_bayes"),
    )


def score_predictions(actual, predicted):
    return {
        "validation_accuracy": accuracy_score(actual, predicted),
        "validation_macro_f1": f1_score(
            actual, predicted, labels=CLASSES, average="macro", zero_division=0,
        ),
    }


if __name__ == "__main__":
    data = load_dataset()
    _, folds = split_by_student(data["id"])
    train_mask, validation_mask = folds[0]
    train, validation = data.loc[train_mask], data.loc[validation_mask]
    X_train, X_validation = prepare_fold(train, validation)
    predicted = MixedNaiveBayes().fit(X_train, train["label"]).predict(X_validation)
    print(f"Fold 1: {len(train)} training rows, {len(validation)} validation rows")
    print(f"Feature columns: {X_train.shape[1]}")
    for metric, value in score_predictions(validation["label"], predicted).items():
        print(f"{metric}: {value:.4f}")
