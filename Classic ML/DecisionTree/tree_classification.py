from collections import Counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeCustom:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def _gini(self, y):
        classes = np.unique(y)
        g = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            g -= p ** 2
        return g

    def _entropy(self, y):
        classes = np.unique(y)
        h = 0.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            h -= p * np.log2(p + 1e-9)
        return h

    def _impurity(self, y):
        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    def _best_split(self, X, y):
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

    def _build_tree(self, X, y, depth: int = 0):
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

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node: Node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
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