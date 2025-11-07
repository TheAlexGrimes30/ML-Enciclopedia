from collections import Counter
from typing import Optional, Union, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node:
    """
    Node class for Decision Tree structure
    A node may represent either:
    - internal node: defined by feature & threshold, with left and right children
    - leaf node: defined only by value
    """

    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
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

class DecisionTreeCustom:
    """
    Custom implementation of Decision Tree Classifier
    methods: constructor, fit, predict
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, criterion: str = "gini"):
        """
        Constructor
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required to split a node
        :param criterion: impurity criterion: "gini" or "entropy"
        :return: None
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculates Gini impurity
        :param y: target class labels
        :return: gini impurity value
        """

        classes = np.unique(y)
        g = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            g -= p ** 2
        return g

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates entropy
        :param y: target class labels
        :return: entropy value
        """

        classes = np.unique(y)
        h = 0.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            h -= p * np.log2(p + 1e-9)
        return h

    def _impurity(self, y: np.ndarray) -> float:
        """
        Select impurity function based on criterion
        :param y: target labels
        :return: impurity value
        """

        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Finds best feature and threshold to split data
        :param X: input features
        :param y: target labels
        :return: best feature index, threshold and gain
        """

        best_gain = 0
        best_feat, best_thresh = None, None
        current_impurity = self._impurity(y)
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

                gain = current_impurity - (p_left * self._impurity(y_left) + (1 - p_left) * self._impurity(y_right))

                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, t

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the decision tree
        :param X: feature matrix
        :param y: target labels
        :param depth: current depth of the tree
        :return: Node
        """

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feat, thresh, gain = self._best_split(X, y)

        if gain == 0:
            leaf_value = Counter().most_common(1)[0][0]
            return Node(value=leaf_value)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit builds the decision tree from training data
        :param X: training input features
        :param y: training target labels
        :return: None
        """

        self.root = self._build_tree(X, y)

    def _predict_one(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """
        Predicts class for a single sample
        :param x: input sample (1D array)
        :param node: current tree node
        :return: predicted class label
        """

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for dataset
        :param X: input features (n_samples, n_features)
        :return: predicted labels array
        """

        return np.array([self._predict_one(x, self.root) for x in X])


X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

custom_tree = DecisionTreeCustom(max_depth=5, criterion='gini')

custom_tree.fit(X_train, y_train)

y_pred_custom = custom_tree.predict(X_test)

sk_tree = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)

sk_tree.fit(X_train, y_train)

y_pred_sk = sk_tree.predict(X_test)

print("Custom Decision Tree accuracy:", accuracy_score(y_test, y_pred_custom))
print("Custom Decision Tree precision:", precision_score(y_test, y_pred_custom))
print("Custom Decision Tree recall:", recall_score(y_test, y_pred_custom))
print("Custom Decision Tree f1 score:", f1_score(y_test, y_pred_custom))
print("Custom Decision Tree roc auc score:", roc_auc_score(y_test, y_pred_custom))
print("\n")
print("Sklearn Decision Tree accuracy:", accuracy_score(y_test, y_pred_sk))
print("Sklearn Decision Tree precision:", precision_score(y_test, y_pred_sk))
print("Sklearn Decision Tree recall:", recall_score(y_test, y_pred_sk))
print("Sklearn Decision Tree f1 score:", f1_score(y_test, y_pred_sk))
print("Sklearn Decision Tree roc auc score:", roc_auc_score(y_test, y_pred_sk))