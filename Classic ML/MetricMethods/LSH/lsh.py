import heapq

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class LSHKNN:
    """
    k-Nearest Neighbors classifier using Locality-Sensitive Hashing (LSH).

    This implementation uses random hyperplanes to hash points into buckets.
    During prediction, only points from the same hash bucket are considered
    as candidate neighbors, which significantly reduces the search space.
    """

    def __init__(self, k: int = 3, n_planes: int = 10):
        """
        Initialize the LSHKNN classifier.

        Parameters
            k : int, default=3
                Number of nearest neighbors to use for classification.

            n_planes : int, default=10
                Number of random hyperplanes used for hashing.
                More planes increase selectivity but reduce recall.
        """

        self.k = k
        self.n_planes = n_planes
        self.hash_tables = {}
        self.planes = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build hash tables for the training data.

        Each data point is projected onto multiple random hyperplanes,
        and a binary hash is generated based on the sign of projections.

        Parameters
            X : np.ndarray
                Training feature matrix of shape (n_samples, n_features).

            y : np.ndarray
                Training labels of shape (n_samples,).
        """

        self.X = X
        self.y = y
        dim = X.shape[1]

        self.planes = np.random.randn(self.n_planes, dim)

        for idx, point in enumerate(X):
            h = self._hash(point)

            if h not in self.hash_tables:
                self.hash_tables[h] = []

            self.hash_tables[h].append(idx)

    def _hash(self, point: np.ndarray) -> tuple:
        """
        Compute the hash of a data point using random projections.

        Each projection determines a binary value:
        - True if projection > 0
        - False otherwise

        The final hash is a tuple of boolean values.

        Parameters
            point : np.ndarray
                Input vector of shape (n_features,).

        Returns
            tuple
                Binary hash representing the bucket key.
        """

        projections = point @ self.planes.T
        return tuple(projections > 0)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.

        For each test point:
        - Compute its hash
        - Retrieve candidate points from the same bucket
        - Perform KNN on the candidate set
        - If no candidates are found, fallback to random prediction

        Parameters
            X_test : np.ndarray
                Test feature matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Predicted class labels.
        """

        predictions = []

        for x in X_test:
            h = self._hash(x)
            candidate_indices = self.hash_tables.get(h, [])
            neighbors = []

            for idx in candidate_indices:
                dist = np.linalg.norm(self.X[idx] - x)

                if len(neighbors) < self.k:
                    heapq.heappush(neighbors, (-dist, self.y[idx]))
                else:
                    if dist < -neighbors[0][0]:
                        heapq.heappop(neighbors)
                        heapq.heappush(neighbors, (-dist, self.y[idx]))

            if neighbors:
                labels = [label for _, label in neighbors]
                prediction = np.bincount(labels).argmax()
            else:
                prediction = np.random.choice(self.y)

            predictions.append(prediction)

        return  np.array(predictions)

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

lsh_knn = LSHKNN(k=3, n_planes=12)

lsh_knn.fit(X_train, y_train)

y_pred_lsh = lsh_knn.predict(X_test)

print("LSH KNN accuracy:", accuracy_score(y_test, y_pred_lsh))
print("LSH KNN precision:", precision_score(y_test, y_pred_lsh))
print("LSH KNN recall:", recall_score(y_test, y_pred_lsh))
print("LSH KNN f1 score:", f1_score(y_test, y_pred_lsh))
print("LSH KNN roc auc score:", roc_auc_score(y_test, y_pred_lsh))
