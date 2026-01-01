import time
from collections import Counter
from itertools import combinations
from typing import Tuple, Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=4,
    n_clusters_per_class=1,
    weights=[0.2, 0.3, 0.25, 0.25],
    flip_y=0.05,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

unique, counts = np.unique(y, return_counts=True)
print("\nРаспределение классов во всей выборке:")
for cls, cnt in zip(unique, counts):
    print(f"  Класс {cls}: {cnt} образцов ({cnt/len(y)*100:.1f}%)")

print()

class BinaryLogisticRegression:
    """
    Logistic regression class
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        """
        Logistic Regression Constructor

        Parameters
            lr : float
                Learning rate for gradient descent.
            n_iters : int
                Number of gradient descent iterations.
        """

        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model using gradient descent.

        Parameters
            X : np.ndarray
                Training feature matrix of shape (n_samples, n_features).
            y : np.ndarray
                Binary target vector of shape (n_samples,).
        """

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iters):
            linear = X @ self.w + self.b
            y_pred = 1 / (1 + np.exp(-linear))

            dw = (1 / len(y)) * X.T @ (y_pred - y)
            db = np.mean(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probabilities for input samples.

        Parameters
            X : np.ndarray
                Feature matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted probabilities for the positive class.
        """

        linear = X @ self.w + self.b
        return 1 / (1 + np.exp(-linear))

class CustomOneVsRest:
    """
    One-vs-Rest multiclass classification
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train one binary classifier per class.

        Parameters
            X : np.ndarray
                Training feature matrix.
            y : np.ndarray
                Multiclass target vector.
        """

        self.classes = np.unique(y)
        self.models = {}

        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            model = BinaryLogisticRegression()
            model.fit(X, y_binary)
            self.models[c] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
            X : np.ndarray
                Feature matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted class labels.
        """

        probs = np.column_stack([
            self.models[c].predict_proba(X)
            for c in self.classes
        ])

        return self.classes[np.argmax(probs, axis=1)]

class CustomOneVsOne:
    """
    One-vs-One multiclass classification
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a binary classifier for each pair of classes.

        Parameters
            X : np.ndarray
                Training feature matrix.
            y : np.ndarray
                Multiclass target vector.
        """

        self.classes = np.unique(y)
        self.models = {}

        for c1, c2 in combinations(self.classes, 2):
            idx = np.logical_or(y == c1, y == c2)
            X_pair = X[idx]
            y_pair = (y[idx] == c1).astype(int)

            model = BinaryLogisticRegression()
            model.fit(X_pair, y_pair)
            self.models[(c1, c2)] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting over pairwise classifiers.

        Parameters
            X : np.ndarray
                Feature matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted class labels.
        """

        predictions = []

        for x in X:
            votes = []
            for (c1, c2), model in self.models.items():
                p = model.predict_proba(x.reshape(1, -1))
                votes.append(c1 if p[0] >= 0.5 else c2)
            predictions.append(Counter(votes).most_common(1)[0][0])

        return np.array(predictions)

def evaluate(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
             y_test: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Train a model, evaluate its performance, and measure training time.

    Parameters
        model
            Classification model with fit and predict methods.
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target vector.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test target vector.

    Returns
        Tuple[Dict[str, np.ndarray], float]
            Dictionary with evaluation metrics and training time.
    """

    metrics = {}
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics["accuracy"] = accuracy

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy

    precision = precision_score(y_test, y_pred, average="weighted")
    metrics["precision"] = precision

    f1 = f1_score(y_test, y_pred, average="weighted")
    metrics["f1"] = f1

    recall = recall_score(y_test, y_pred, average="weighted")
    metrics["recall"] = recall

    return metrics, train_time

results = {}

results["Custom OvO"] = evaluate(
    CustomOneVsOne(), X_train, y_train, X_test, y_test
)

results["Custom OvR"] = evaluate(
    CustomOneVsRest(), X_train, y_train, X_test, y_test
)

results["Sklearn OvO"] = evaluate(
    OneVsOneClassifier(LogisticRegression(max_iter=1000)),
    X_train, y_train, X_test, y_test
)

results["Sklearn OvR"] = evaluate(
    OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    X_train, y_train, X_test, y_test
)

results["Multinomial LR"] = evaluate(
    LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    ),
    X_train, y_train, X_test, y_test
)

for name, (metrics, t) in results.items():
    print(f"{name:<20}")
    print(f"Time: {t:<20}")

    for metric_name, metrics_value in metrics.items():
        print(f"{metric_name}: {metrics_value:<20}")

    print("=" * 50)
    print()
