import heapq

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class LSHKNN:
    def __init__(self, k: int = 3, n_planes: int = 10):
        self.k = k
        self.n_planes = n_planes
        self.hash_tables = {}
        self.planes = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        projections = point @ self.planes.T
        return tuple(projections > 0)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
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
