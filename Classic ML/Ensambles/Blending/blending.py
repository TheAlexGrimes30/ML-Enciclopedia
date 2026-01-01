from typing import Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    weights=[0.2, 0.3, 0.25, 0.25],
    flip_y=0.05,
    random_state=42
)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

X_train, X_blend, y_train, y_blend = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full,
    random_state=42
)

base_models = {
    "logreg": LogisticRegression(
        max_iter=1000
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "svm": SVC(
        probability=True,
        kernel="rbf"
    )
}

for model in base_models.values():
    model.fit(X_train, y_train)

def get_blending_features(models, X: np.ndarray) -> np.ndarray:
    """
    Generate meta-features for blending by concatenating class probability
    predictions from multiple base models.

    Each base model produces a probability distribution over classes for
    each sample. These probability vectors are horizontally stacked to
    form a new feature representation that is used as input for the
    meta-model in the blending ensemble.

    Parameters
    models : dict
        Dictionary of trained base models. Each model must implement
        the `predict_proba` method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
        np.ndarray
            Meta-feature matrix of shape (n_samples, n_classes * n_models),
            where each block of features corresponds to class probabilities
            predicted by one base model.
    """
    return np.hstack([
        model.predict_proba(X)
        for model in models.values()
    ])

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute classification performance metrics for model evaluation.

    This function calculates a set of commonly used classification metrics
    based on the true labels and the predicted labels. All metrics are
    computed using a weighted averaging strategy to account for possible
    class imbalance.

    Parameters
        y_true : np.ndarray
            Ground truth class labels of shape (n_samples,).
        y_pred : np.ndarray
            Predicted class labels of shape (n_samples,).

    Returns
        Dict[str, np.ndarray]
            Dictionary containing the following evaluation metrics:
            - accuracy: Overall classification accuracy
            - balanced_accuracy: Average recall obtained on each class
            - precision: Weighted precision score
            - recall: Weighted recall score
            - f1: Weighted F1-score
    """
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

X_blend_meta = get_blending_features(base_models, X_blend)
X_test_meta = get_blending_features(base_models, X_test)

results = {}

meta_model = LogisticRegression(
    max_iter=1000
)

meta_model.fit(X_blend_meta, y_blend)
y_pred_blend = meta_model.predict(X_test_meta)

for name, model in base_models.items():
    y_pred = model.predict(X_test)
    results[name] = evaluate(y_test, y_pred)

results["blending"] = evaluate(y_test, y_pred_blend)

for name, metrics in results.items():
    print(f"{name.upper():<15}")
    for metric, value in metrics.items():
        print(f"{metric:<20}: {value:.4f}")
    print("-" * 40)
