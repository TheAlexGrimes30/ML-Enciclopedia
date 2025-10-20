import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor:
    def __init__(self, k: int = 2):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, 1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_targets = self.y_train[k_indices]
            prediction = np.mean(k_nearest_targets)
            predictions.append(prediction)
        return np.array(predictions)

Xr, yr = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

my_knn_reg = KNNRegressor(k=5)
my_knn_reg.fit(Xr_train, yr_train)
yr_pred_my = my_knn_reg.predict(Xr_test)

sk_knn_reg = KNeighborsRegressor(n_neighbors=5)
sk_knn_reg.fit(Xr_train, yr_train)
yr_pred_sk = sk_knn_reg.predict(Xr_test)

print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my))
print("Sklearn KNNRegressor R2:", r2_score(yr_test, yr_pred_sk))
