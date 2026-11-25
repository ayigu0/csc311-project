"""Softmax regression model experiments."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_exploration.dataexploration import load_dataset, split_by_student
from data_exploration.preprocessing import SurveyPreprocessor


CLASSES = ("Baseball", "Basketball", "Soccer")
CLASS_TO_INDEX = {sport: index for index, sport in enumerate(CLASSES)}
MAX_EPOCHS = 1000
PATIENCE = 20
RELATIVE_IMPROVEMENT = 0.0001  # 0.01%


def _encode_labels(labels):
    encoded = labels.map(CLASS_TO_INDEX)
    if encoded.isna().any():
        raise ValueError("Expected Baseball, Basketball, or Soccer in the label column.")
    return encoded.astype("int64")


def prepare_fold(data, fold_number=1):
    """Return X_train, X_val, y_train, y_val for a shared development fold.

    Folds are numbered starting at 1. Features are DataFrames and labels are
    integer Series with matching row indexes.
    """
    _, folds = split_by_student(data["id"])
    if not 1 <= fold_number <= len(folds):
        raise ValueError(f"Fold number must be between 1 and {len(folds)}.")

    train_mask, val_mask = folds[fold_number - 1]
    train = data.loc[train_mask]
    val = data.loc[val_mask]
    y_train = _encode_labels(train["label"])
    y_val = _encode_labels(val["label"])

    preprocessor = SurveyPreprocessor().fit(train)
    X_train = preprocessor.transform(train, model="softmax")
    X_val = preprocessor.transform(val, model="softmax")
    return X_train, X_val, y_train, y_val


def _shifted_scores(features, weights, bias):
    scores = features @ weights + bias
    # Subtracting the row maximum keeps the exponentials in a safe range.
    return scores - scores.max(axis=1, keepdims=True)


def _training_loss(features, labels, weights, bias, l2):
    scores = _shifted_scores(features, weights, bias)
    log_normalizer = np.log(np.exp(scores).sum(axis=1))
    cross_entropy = (log_normalizer - scores[np.arange(len(labels)), labels]).mean()
    return float(cross_entropy + 0.5 * l2 * np.sum(weights**2))


class SoftmaxRegression:
    """Fit multinomial softmax with mini-batch gradient descent and L2."""

    def __init__(self, learning_rate=0.01, batch_size=32, l2=0.001, seed=42):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if features.ndim != 2 or labels.shape != (len(features),) or not len(features):
            raise ValueError("Expected a nonempty feature matrix and one label per row.")
        if not np.isfinite(features).all():
            raise ValueError("Training features must be finite.")
        if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, range(len(CLASSES))).all():
            raise ValueError("Expected integer class indices 0, 1, or 2.")

        rng = np.random.default_rng(self.seed)
        weights = np.zeros((features.shape[1], len(CLASSES)))
        bias = np.zeros(len(CLASSES))
        reference_loss = _training_loss(features, labels, weights, bias, self.l2)
        self.loss_history_ = [reference_loss]  # First entry is the untrained loss.
        best_loss = np.inf
        stale_epochs = 0
        self.stop_reason_ = "epoch limit"

        for epoch in range(1, MAX_EPOCHS + 1):
            order = rng.permutation(len(labels))
            for start in range(0, len(labels), self.batch_size):
                batch = order[start:start + self.batch_size]
                X_batch, y_batch = features[batch], labels[batch]

                errors = np.exp(_shifted_scores(X_batch, weights, bias))
                errors /= errors.sum(axis=1, keepdims=True)
                errors[np.arange(len(batch)), y_batch] -= 1.0
                errors /= len(batch)

                weight_gradient = X_batch.T @ errors + self.l2 * weights
                bias_gradient = errors.sum(axis=0)
                weights -= self.learning_rate * weight_gradient
                bias -= self.learning_rate * bias_gradient

            loss = _training_loss(features, labels, weights, bias, self.l2)
            if not np.isfinite(loss):
                raise FloatingPointError(f"Training loss became non-finite at epoch {epoch}.")
            self.loss_history_.append(loss)

            # Keep the lowest-loss checkpoint even if its improvement is too
            # small to reset patience.
            if loss < best_loss:
                best_loss = loss
                best_weights, best_bias = weights.copy(), bias.copy()
                self.best_epoch_ = epoch

            if reference_loss - loss >= RELATIVE_IMPROVEMENT * reference_loss:
                reference_loss = loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs == PATIENCE:
                    self.stop_reason_ = "training loss plateau"
                    break

        self.weights_, self.bias_ = best_weights, best_bias
        self.n_epochs_ = epoch
        return self

    def predict(self, X):
        """Return the class index with the highest score for each row."""
        scores = np.asarray(X, dtype=float) @ self.weights_ + self.bias_
        return scores.argmax(axis=1)


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_fold(load_dataset())
    print(f"Fold 1 training: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Fold 1 validation: {X_val.shape[0]} rows, {X_val.shape[1]} features")

    model = SoftmaxRegression(learning_rate=0.01, batch_size=32, l2=0.001)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    print(predictions)

    print(f"Epochs run: {model.n_epochs_} ({model.stop_reason_})")
    print(f"Training loss before fitting: {model.loss_history_[0]:.6f}")
    print(f"Best training loss: {model.loss_history_[model.best_epoch_]:.6f} at epoch {model.best_epoch_}")
    print(f"Validation accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Validation macro F1: {f1_score(y_val, predictions, labels=range(len(CLASSES)), average='macro'):.4f}")
