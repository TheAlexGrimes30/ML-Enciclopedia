import heapq

import numpy as np
from networkx.classes import neighbors
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree


class KDNode:
    """
    KD-tree node representation.

    Attributes
        point : np.ndarray
            Feature vector of the point stored in the node.
        label : int
            Class label corresponding to the point.
        axis : int
            Axis (dimension) used for splitting at this node.
        left : KDNode or None
            Left subtree (points with smaller coordinate along axis).
        right : KDNode or None
            Right subtree (points with larger coordinate along axis).
    """

    def __init__(
            self,
            point: np.ndarray,
            label: int,
            axis: int,
            left: "KDNode" = None,
            right: "KDNode" = None
    ):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTreeKNN:
    """
    k-Nearest Neighbors classifier based on KD-tree.

    This classifier builds a KD-tree from training data and uses it
    to efficiently search for k nearest neighbors during prediction.
    """

    def __init__(self, k: int = 3):
        """
        Constructor.

        Parameters
            k : int
                Number of nearest neighbors to consider.
        """

        self.k = k
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build KD-tree from training data.

        Parameters
            X : np.ndarray
                Training samples of shape (n_samples, n_features).
            y : np.ndarray
                Target labels of shape (n_samples,).

        Returns
            None
        """

        data = list(zip(X, y))
        self.root = self._build_tree(data, depth=0)

    def _build_tree(self, data: list, depth: int) -> KDNode:
        """
        Recursively build KD-tree.

        Parameters
            data : list
                List of tuples (point, label).
            depth : int
                Current depth in the tree.

        Returns
            KDNode or None
                Root node of the subtree.
        """

        if not data:
            return None

        axis = depth % len(data[0][0])
        data.sort(key=lambda  x: x[0][axis])
        median = len(data) // 2

        return KDNode(
            point=data[median][0],
            label=data[median][1],
            axis=axis,
            left=self._build_tree(data[:median], depth + 1),
            right=self._build_tree(data[median + 1:], depth + 1)
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.

        Parameters
            X_test : np.ndarray
                Test samples of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted class labels.
        """

        predictions = []

        for x in X_test:
            neighbors = []
            self._knn_search(self.root, x, neighbors)
            labels = [label for _, label in neighbors]
            predictions.append(np.bincount(labels).argmax())

        return np.array(predictions)

    def _knn_search(
            self,
            node: KDNode,
            target:np.ndarray,
            neighbors: list) -> None:

        """
        Perform kNN search in KD-tree.

        Parameters
            node : KDNode
                Current node in KD-tree.
            target : np.ndarray
                Target point.
            neighbors : list
                Max-heap storing current nearest neighbors.

        Returns
            None
        """

        if node is None:
            return

        dist = np.linalg.norm(node.point - target)

        if len(neighbors) < self.k:
            heapq.heappush(neighbors, (-dist, node.label))
        else:
            if dist < -neighbors[0][0]:
                heapq.heappop(neighbors)
                heapq.heappush(neighbors, (-dist, node.label))

        axis = node.axis
        diff = target[axis] - node.point[axis]

        close_branch = node.left if diff < 0 else node.right
        far_branch = node.right if diff < 0 else node.left

        self._knn_search(close_branch, target, neighbors)

        if len(neighbors) < self.k or abs(diff) < -neighbors[0][0]:
            self._knn_search(far_branch, target, neighbors)

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sk_kdtree = KDTree(X_train)
dist, ind = sk_kdtree.query(X_test, k=3)

y_pred_sk_kdtree = np.array([
    np.bincount(y_train[i]).argmax() for i in ind
])

kd_knn = KDTreeKNN(k=3)
kd_knn.fit(X_train, y_train)
y_pred_kd = kd_knn.predict(X_test)

print("Custom KDTree KNN accuracy:", accuracy_score(y_test, y_pred_kd))
print("Custom KDTree KNN precision:", precision_score(y_test, y_pred_kd))
print("Custom KDTree KNN recall:", recall_score(y_test, y_pred_kd))
print("Custom KDTree KNN f1 score:", f1_score(y_test, y_pred_kd))
print("Custom KDTree KNN roc auc score:", roc_auc_score(y_test, y_pred_kd))

print("\nSklearn KDTree KNN accuracy:", accuracy_score(y_test, y_pred_sk_kdtree))
print("Sklearn KDTree KNN precision:", precision_score(y_test, y_pred_sk_kdtree))
print("Sklearn KDTree KNN recall:", recall_score(y_test, y_pred_sk_kdtree))
print("Sklearn KDTree KNN f1 score:", f1_score(y_test, y_pred_sk_kdtree))
print("Sklearn KDTree KNN roc auc score:", roc_auc_score(y_test, y_pred_sk_kdtree))
