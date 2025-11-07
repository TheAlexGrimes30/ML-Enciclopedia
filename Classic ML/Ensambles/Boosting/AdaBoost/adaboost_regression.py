from typing import Tuple, Optional, Union

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

class Node:
    """
    Node class for Decision Tree structure
    A node may represent either:
    - internal node: defined by feature & threshold, with left and right children
    - leaf node: defined only by value
    """

    def __init__(self, feature: Optional[int] = None, threshold: Optional[np.floating] = None,
                 left: Optional["Node"] = None, right: Optional["Node"] = None,
                 value: Optional[Union[int, float]] = None):
        """
        Constructor
        :param feature: index of feature used for split
        :param threshold: threshold value for split
        :param left: left child node
        :param right: right child node
        :param value: predicted class for leaf node
        :return: None
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressorCustom:
    """
    Custom Decision Tree Regressor implementation (CART algorithm)
    Uses MSE (mean squared error) to evaluate splits.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _mse(self, y: np.ndarray) -> np.floating:
        """
        Computes Mean Squared Error (impurity measure for regression)
        :param y: target values
        :return: MSE value
        """

        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Finds the best feature and threshold to split data minimizing MSE
        :param X: features matrix (n_samples, n_features)
        :param y: target values (n_samples,)
        :return: (best_feature_index, best_threshold_value, best_gain)
        """

        best_gain = 0
        best_feat, best_thresh = 0, 0
        current_mse = self._mse(y)
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])

            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                p_left = len(y_left) / n_samples
                gain = current_mse - (p_left * self._mse(y_left) + (1 - p_left) * self._mse(y_right))

                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, t

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds decision tree
        :param X: features matrix
        :param y: target values
        :param depth: current depth of recursion
        :return: Node object (leaf or internal)
        """

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))

        feat, thresh, gain = self._best_split(X, y)
        if gain == 0:
            return Node(value=np.mean(y))

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Decision Tree model to training data
        :param X: training features
        :param y: training target values
        :return: None
        """

        self.root = self._build_tree(X, y)

    def _predict_one(self, x: np.ndarray, node: Node) -> int | float | None:
        """
        Predicts value for a single sample
        :param x: feature vector
        :param node: current node
        :return: predicted continuous value
        """

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts values for dataset
        :param X: features matrix
        :return: predicted values (n_samples,)
        """

        return np.array([self._predict_one(x, self.root) for x in X])

class CustomAdaBoostRegressor:
    """
    Custom AdaBoost regressor implementation.

    This ensemble method trains multiple weak regressors sequentially,
    adjusting sample weights to focus on poorly predicted samples.
    Each model's prediction is weighted according to its accuracy.
    """

    def __init__(self, base_estimator: DecisionTreeRegressorCustom, n_estimators: int = 10):
        """
        Constructor

        :param base_estimator: base regressor class (not an instance), e.g., DecisionTreeRegressorCustom
        :param n_estimators: number of weak learners in the ensemble
        :return: None
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.betas = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train AdaBoost regressor ensemble.

        :param X: feature matrix, shape (n_samples, n_features)
        :param y: target values, shape (n_samples,)
        :return: None
        """

        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = self.base_estimator()
            model.fit(X, y)
            y_pred = model.predict(X)

            errors = np.abs(y - y_pred)
            max_error = np.max(errors)
            if max_error == 0:
                break

            rel_errors = errors / max_error
            err_m = np.dot(w, rel_errors)
            if err_m >= 0.5:
                break

            beta = err_m / (1 - err_m)

            w *= np.power(beta, (1 - rel_errors))
            w /= np.sum(w)

            self.models.append(model)
            self.betas.append(beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using weighted average of weak learners.

        :param X: feature matrix to predict on, shape (n_samples, n_features)
        :return: predicted target values, shape (n_samples,)
        """

        model_preds = np.array([model.predict(X) for model in self.models])
        weights = np.log(1 / np.array(self.betas))
        y_pred = np.average(model_preds, axis=0, weights=weights)
        return y_pred

X, y = make_regression(n_samples=500, n_features=5, n_informative=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

adaboost_custom = CustomAdaBoostRegressor(base_estimator=lambda: DecisionTreeRegressorCustom(max_depth=5), n_estimators=10)
adaboost_custom.fit(X_train, y_train)
y_pred_custom = adaboost_custom.predict(X_test)

adaboost_sklearn = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5),
                               n_estimators=10, random_state=42)

adaboost_sklearn.fit(X_train, y_train)
y_pred_sklearn = adaboost_sklearn.predict(X_test)

print("Custom AdaBoost MSE:", mean_squared_error(y_test, y_pred_custom))
print("Custom AdaBoost R2:", r2_score(y_test, y_pred_custom))
print("\n")
print("Sklearn AdaBoost MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("Sklearn AdaBoost R2:", r2_score(y_test, y_pred_sklearn))

