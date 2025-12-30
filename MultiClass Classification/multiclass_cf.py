import time
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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

class BinaryLogisticRegression:
    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        linear = X @ self.w + self.b
        return 1 / (1 + np.exp(-linear))

class CustomOneVsRest:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        self.models = {}

        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            model = BinaryLogisticRegression()
            model.fit(X, y_binary)
            self.models[c] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = np.column_stack([
            self.models[c].predict_proba(X)
            for c in self.classes
        ])

        return self.classes[np.argmax(probs, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([
            self.models[c].predict_proba(X)
            for c in self.classes
        ])

class CustomOneVsOne:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        self.models = {}

        for c1, c2 in combinations(self.classes, 2):
            idx = np.logical_or(y == c1, y == c2)
            X_pair = X[idx]
            y_pair = (y[idx] == c1).astype(int)

            model = BinaryLogisticRegression()
            model.fit(X_pair, y_pair)
            self.models[(c1, c2)] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        votes = np.zeros((n_samples, n_classes))
        for (c1, c2), model in self.models.items():
            proba_pair = model.predict_proba(X)

            idx_c1 = np.where(self.classes == c1)[0][0]
            idx_c2 = np.where(self.classes == c2)[0][0]

            votes[:, idx_c1] += (1 - proba_pair)
            votes[:, idx_c2] += proba_pair

        row_sums = votes.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        proba = votes / row_sums[:, np.newaxis]

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for x in X:
            votes = []
            for (c1, c2), model in self.models.items():
                p = model.predict_proba(x.reshape(1, -1))
                votes.append(c1 if p[0] >= 0.5 else c2)
            predictions.append(Counter(votes).most_common(1)[0][0])

        return np.array(predictions)

def evaluate(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
             y_test: np.ndarray, proba: bool = True):

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if proba:
        y_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_proba)
    else:
        loss = None

    return acc, train_time

results = {}

results["Custom OvO"] = evaluate(
    CustomOneVsOne(), X_train, y_train, X_test, y_test
)

results["Custom OvR"] = evaluate(
    CustomOneVsRest(), X_train, y_train, X_test, y_test
)

results["Sklearn OvO"] = evaluate(
    OneVsOneClassifier(LogisticRegression(max_iter=1000)),
    X_train, y_train, X_test, y_test,
    proba=False
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

for name, (acc, t) in results.items():
    print(f"{name:<25} {acc:<12} {t:<12}")
