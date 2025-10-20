import numpy as np


class KNNRegression:
    def __init__(self, k: int = 2):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, 1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_targets = self.X_train[k_indices]
            prediction = np.mean(k_nearest_targets)
            predictions.append(prediction)
        return np.array(predictions)
