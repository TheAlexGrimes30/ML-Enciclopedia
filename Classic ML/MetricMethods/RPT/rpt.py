import heapq
from typing import Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class RPTreeNode:
    def __init__(
            self,
            vector: Optional[np.ndarray] = None,
            threshold: Optional[float] = None,
            points: Optional[np.ndarray] = None,
            labels: Optional[np.ndarray] = None,
            left: Optional["RPTreeNode"] = None,
            right: Optional["RPTreeNode"] = None
    ):
        """
        Random Projection Tree node.

        Attributes
            vector : np.ndarray
                Random projection vector.
            threshold : float
                Projection threshold used for splitting.
            points : np.ndarray
                Points stored in leaf node.
            labels : np.ndarray
                Labels stored in leaf node.
            left : RPTreeNode
                Left subtree.
            right : RPTreeNode
                Right subtree.
            """

        self.vector = vector
        self.threshold = threshold
        self.points = points
        self.labels = labels
        self.left = left
        self.right = right

class RPTreeKNN:
    """
    k-Nearest Neighbors classifier using Random Projection Trees.

    This algorithm builds a tree by recursively splitting the data using
    random projection directions. It is particularly useful in high-dimensional
    spaces where KD-Tree performance degrades.

    The tree enables approximate nearest neighbor search by exploring
    promising branches and pruning unlikely regions.
    """

    def __init__(self, k: int = 3, leaf_size: int = 10):
        """
        Initialize the RPTreeKNN classifier.

        Parameters
            k : int, default=3
                Number of nearest neighbors used for prediction.

            leaf_size : int, default=10
                Maximum number of points stored in a leaf node.
        """

        self.k = k
        self.leaf_size = leaf_size
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the Random Projection Tree from training data.

        Parameters
            X : np.ndarray
                Training feature matrix of shape (n_samples, n_features).

            y : np.ndarray
                Training labels of shape (n_samples,).
        """

        self.root = self._build_tree(X, y)

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> "RPTreeNode":
        """
        Recursively construct the Random Projection Tree.

        At each node:
            - A random projection vector is generated
            - Data is projected onto this vector
            - Points are split based on the median projection value

        Parameters
            X : np.ndarray
                Subset of feature vectors.

            y : np.ndarray
                Corresponding labels.

        Returns
            RPTreeNode
                Root node of the constructed subtree.
        """

        if len(X) <= self.leaf_size:
            return RPTreeNode(points=X, labels=y)

        dim = X.shape[1]

        random_vector = np.random.randn(dim)
        projections = X @ random_vector
        threshold = np.median(projections)

        left_mask = projections < threshold
        right_mask = projections >= threshold

        left = self._build_tree(X[left_mask], y[left_mask])
        right = self._build_tree(X[right_mask], y[right_mask])

        return RPTreeNode(
            vector=random_vector,
            threshold=threshold,
            left=left,
            right=right
        )

    def _search(self, node: "RPTreeNode", target: np.ndarray, neighbors: list) -> None:
        """
        Search for k nearest neighbors in the Random Projection Tree.

        The algorithm:
        - Traverses the tree based on projection of the target point
        - Searches the most promising branch first
        - Optionally explores the opposite branch if needed

        Parameters
            node : Optional[RPTreeNode]
                Current node in the tree.

            target : np.ndarray
                Query point.

            neighbors : list
                Max-heap storing current nearest neighbors as (-distance, label).
        """

        if node is None:
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

        projection = target @ node.vector

        if projection < node.threshold:
            close_branch = node.left
            far_branch = node.right
        else:
            close_branch = node.right
            far_branch = node.left

        self._search(close_branch, target, neighbors)

        if len(neighbors) < self.k or abs(projection - node.threshold) < -neighbors[0][0]:
            self._search(far_branch, target, neighbors)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.

        For each test point:
        - Perform tree search to find k nearest neighbors
        - Use majority voting to determine the predicted label

        Parameters
            X_test : np.ndarray
                Test feature matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted class labels.
        """

        predictions = []

        for x in X_test:
            neighbors = []
            self._search(self.root, x, neighbors)

            labels = [label for _, label in neighbors]

            predictions.append(np.bincount(labels).argmax())

        return np.array(predictions)

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

rp_knn = RPTreeKNN(k=3)

rp_knn.fit(X_train, y_train)

y_pred_rp = rp_knn.predict(X_test)

print("Random Projection Tree KNN accuracy:", accuracy_score(y_test, y_pred_rp))
print("Random Projection Tree KNN precision:", precision_score(y_test, y_pred_rp))
print("Random Projection Tree KNN recall:", recall_score(y_test, y_pred_rp))
print("Random Projection Tree KNN f1 score:", f1_score(y_test, y_pred_rp))
print("Random Projection Tree KNN roc auc score:", roc_auc_score(y_test, y_pred_rp))