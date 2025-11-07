from typing import Optional, Union, Tuple

import numpy as np
from numpy import floating
from sklearn.datasets import make_regression
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

    def __init__(self, feature: Optional[int] = None, threshold: Optional[floating] = None,
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

    def _mse(self, y: np.ndarray) -> floating:
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


X, y = make_regression(n_samples=500, n_features=1, noise=10.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

custom_tree = DecisionTreeRegressorCustom(max_depth=5)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)

sk_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)
y_pred_sk = sk_tree.predict(X_test)

print("Custom Decision Tree MSE:", mean_squared_error(y_test, y_pred_custom))
print("Custom Decision Tree R2:", r2_score(y_test, y_pred_custom))
print("\n")
print("Custom Decision Tree MSE:", mean_squared_error(y_test, y_pred_sk))
print("Sklearn Decision Tree R2:", r2_score(y_test, y_pred_sk))