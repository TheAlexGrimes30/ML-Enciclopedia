from typing import List, Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    weights=[0.2, 0.3, 0.25, 0.25],
    flip_y=0.05,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

base_models = [
    ("logreg", LogisticRegression(max_iter=1000)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("svm", SVC(probability=True))
]

meta_model = LogisticRegression(
    max_iter=1000
)

class CustomStackingClassifier:
    def __init__(self,
                 base_models: List[tuple],
                 meta_model,
                 n_folds: int = 5
    ):

        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_models = len(self.base_models)

        self.oof_predictions_ = np.zeros(
            (n_samples, n_classes * n_models)
        )

        self.fitted_base_models_ = []

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=42
        )

        for model_idx, (_, model) in enumerate(self.base_models):
            oof_preds = np.zeros((n_samples, n_classes))

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                cloned_model = model.__class__(**model.get_params())
                cloned_model.fit(X_train_fold, y_train_fold)
                oof_preds[val_idx] = cloned_model.predict_proba(X_val_fold)

            self.oof_predictions_[
            :, model_idx * n_classes:(model_idx + 1) * n_classes
            ] = oof_preds

            model.fit(X, y)
            self.fitted_base_models_.append(model)

        self.meta_model.fit(self.oof_predictions_, y)

        return self

    def predict(self, X: np.ndarray):
        meta_features = np.hstack([
            model.predict_proba(X)
            for model in self.fitted_base_models_
        ])
        return self.meta_model.predict(meta_features)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

custom_stacking = CustomStackingClassifier(
    base_models=base_models,
    meta_model=meta_model,
    n_folds=5
)

custom_stacking.fit(X_train, y_train)
y_pred_custom = custom_stacking.predict(X_test)

custom_metrics = evaluate(y_test, y_pred_custom)

sklearn_stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        max_iter=1000
    ),
    stack_method="predict_proba",
    cv=5
)

sklearn_stacking.fit(X_train, y_train)
y_pred_sklearn = sklearn_stacking.predict(X_test)

sklearn_metrics = evaluate(y_test, y_pred_sklearn)

results = {}

for name, model in base_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = evaluate(y_test, y_pred)

results["custom_stacking"] = custom_metrics
results["sklearn_stacking"] = sklearn_metrics

for name, metrics in results.items():
    print(f"{name.upper():<20}")
    for metric, value in metrics.items():
        print(f"{metric:<20}: {value:.4f}")
    print("-" * 45)
