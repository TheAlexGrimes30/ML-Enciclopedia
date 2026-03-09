import heapq
from typing import Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree


class BallNode:
    """
    Ball-tree node representation.

    Attributes
        center : np.ndarray
            Center of the hypersphere.
        radius : float
            Radius of the hypersphere.
        points : np.ndarray
            Points contained in the node (for leaf nodes).
        labels : np.ndarray
            Labels of the points.
        left : BallNode
            Left subtree.
        right : BallNode
            Right subtree.
    """

    def __init__(
            self,
            center: np.ndarray,
            radius: float,
            points: Optional[np.ndarray] = None,
            labels: Optional[np.ndarray] = None,
            left: Optional["BallNode"] = None,
            right: Optional["BallNode"] = None
    ):
        self.center = center
        self.radius = radius
        self.points = points
        self.labels = labels
        self.left = left
        self.right = right

class BallTreeKNN:
    def __init__(self, k: int = 3, leaf_size: int = 10):
        self.k = k
        self.leaf_size = leaf_size
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._build_tree(X, y)

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> "BallNode":
        center = np.mean(X, axis=0)
        radius = np.max(np.linalg.norm(X - center, axis=1))

        if len(X) <= self.leaf_size:
            return BallNode(center, radius, X, y)

        variances = np.var(X, axis=0)
        split_axis = np.argmax(variances)

        sorted_idx = np.argsort(X[:, split_axis])
        median = len(X) // 2

        left = self._build_tree(X[sorted_idx[:median]], y[sorted_idx[:median]])
        right = self._build_tree(X[sorted_idx[median:]], y[sorted_idx[median:]])

        return BallNode(center, radius, None, None, left, right)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []

        for x in X_test:
            neighbors = []
            self._search(self.root, x, neighbors)

            labels = [label for _, label in neighbors]
            predictions.append(np.bincount(labels).argmax())

        return np.array(predictions)

    def _search(self, node: "BallNode", target: np.ndarray, neighbors: np.ndarray):
        if node is None:
            return

        dist_to_center = np.linalg.norm(target - node.center)

        if len(neighbors) == self.k and dist_to_center - node.radius > -neighbors[0][0]:
            return

        if node.points is not None:
            for point, label in zip(node.points, node.labels):
                dist = np.linalg.norm(point - target)

                if len(neighbors) < self.k:
                    heapq.heappush(neighbors, (-dist, label))
                else:
                    if dist < -neighbors[0][0]:
                        heapq.heappop(neighbors)
                        heapq.heappush(neighbors, (-dist, label))
            return

        self._search(node.left, target, neighbors)
        self._search(node.right, target, neighbors)


X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

sk_balltree = BallTree(X_train)
dist, ind = sk_balltree.query(X_test, k=3)

y_pred_sk_ball = np.array([
    np.bincount(y_train[i]).argmax() for i in ind
])

ball_knn = BallTreeKNN(k=3)
ball_knn.fit(X_train, y_train)

y_pred_ball = ball_knn.predict(X_test)

print("Custom BallTree KNN accuracy:", accuracy_score(y_test, y_pred_ball))
print("Custom BallTree KNN precision:", precision_score(y_test, y_pred_ball))
print("Custom BallTree KNN recall:", recall_score(y_test, y_pred_ball))
print("Custom BallTree KNN f1 score:", f1_score(y_test, y_pred_ball))
print("Custom BallTree KNN roc auc score:", roc_auc_score(y_test, y_pred_ball))

print("\nSklearn BallTree KNN accuracy:", accuracy_score(y_test, y_pred_sk_ball))
print("Sklearn BallTree KNN precision:", precision_score(y_test, y_pred_sk_ball))
print("Sklearn BallTree KNN recall:", recall_score(y_test, y_pred_sk_ball))
print("Sklearn BallTree KNN f1 score:", f1_score(y_test, y_pred_sk_ball))
print("Sklearn BallTree KNN roc auc score:", roc_auc_score(y_test, y_pred_sk_ball))
