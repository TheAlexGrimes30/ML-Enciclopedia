import heapq
from typing import List

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class HNSWNode:
    """
        Node in HNSW (Hierarchical Navigable Small World) graph.

    Each node represents a data point and maintains connections
    (edges) to its nearest neighbors in the graph.

    Attributes
        vector : np.ndarray
            Feature vector representing the node.

    label : int
        Class label associated with the node.

    neighbors : list[HNSWNode]
        List of neighboring nodes (graph edges).
    """

    def __init__(self, vector: np.ndarray, label: int):
        """
        Initialize an HNSW node.

        Parameters
            vector : np.ndarray
                Feature vector of shape (n_features,).

            label : int
                Class label.
        """

        self.vector = vector
        self.label = label
        self.neighbors = []

class HNSWKNN:
    """
    k-Nearest Neighbors classifier using a simplified HNSW graph.

    HNSW (Hierarchical Navigable Small World) is a graph-based
    approximate nearest neighbor algorithm. It builds a graph where
    nodes are connected to their nearest neighbors, enabling efficient
    navigation during search.
    """

    def __init__(self, k=3, M=5):
        """
        Initialize the HNSWKNN classifier.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors used for prediction.

        M : int, default=5
            Maximum number of neighbors per node (graph connectivity).
        """

        self.k = k
        self.M = M
        self.nodes = []
        self.entry_point = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the HNSW graph from training data.

        For each data point:
        - A node is created
        - The node is connected to its nearest neighbors
        - Graph connectivity is limited by parameter M

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).

        y : np.ndarray
            Training labels of shape (n_samples,).
        """

        for vector, label in zip(X, y):

            node = HNSWNode(vector, label)

            if self.entry_point is None:

                self.entry_point = node
                self.nodes.append(node)
                continue

            neighbors = self._search_layer(vector)

            for n in neighbors[:self.M]:

                node.neighbors.append(n)
                n.neighbors.append(node)

            self.nodes.append(node)

    def _search_layer(self, query: np.ndarray) -> List["HNSWNode"]:
        """
        Perform a naive nearest neighbor search over all nodes.

        This method computes distances from the query to all nodes
        and returns nodes sorted by distance.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape (n_features,).

        Returns
        -------
        list[HNSWNode]
            List of nodes sorted by increasing distance to the query.
        """

        candidates = []

        for node in self.nodes:

            dist = np.linalg.norm(node.vector - query)

            heapq.heappush(candidates, (dist, node))

        candidates.sort(key=lambda x: x[0])

        return [node for _, node in candidates]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.

        For each test point:
        - Perform nearest neighbor search in the graph
        - Select k closest nodes
        - Use majority voting to determine the predicted label

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """

        predictions = []

        for x in X_test:

            neighbors = self._search_layer(x)[:self.k]

            labels = [n.label for n in neighbors]

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


hnsw_knn = HNSWKNN(k=3, M=5)

hnsw_knn.fit(X_train, y_train)

y_pred_hnsw = hnsw_knn.predict(X_test)

print("HNSW KNN accuracy:", accuracy_score(y_test, y_pred_hnsw))
print("HNSW KNN precision:", precision_score(y_test, y_pred_hnsw))
print("HNSW KNN recall:", recall_score(y_test, y_pred_hnsw))
print("HNSW KNN f1 score:", f1_score(y_test, y_pred_hnsw))
print("HNSW KNN roc auc score:", roc_auc_score(y_test, y_pred_hnsw))