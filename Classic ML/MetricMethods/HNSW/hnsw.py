import heapq

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class HNSWNode:
    def __init__(self, vector: np.ndarray, label: int):
        self.vector = vector
        self.label = label
        self.neighbors = []

class HNSWKNN:
    """
    k-Nearest Neighbors classifier using simplified HNSW graph.
    """

    def __init__(self, k=3, M=5):

        self.k = k
        self.M = M
        self.nodes = []
        self.entry_point = None

    def fit(self, X, y):

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

    def _search_layer(self, query):

        candidates = []

        for node in self.nodes:

            dist = np.linalg.norm(node.vector - query)

            heapq.heappush(candidates, (dist, node))

        candidates.sort(key=lambda x: x[0])

        return [node for _, node in candidates]

    def predict(self, X_test):

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