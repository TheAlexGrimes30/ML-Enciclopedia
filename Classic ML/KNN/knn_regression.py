import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor:
    """
    KNN Regression class
    methods: constructor, fit, predict
    """

    def __init__(self, k: int = 2):
        """
        Constructor
        :param k: number of neighbours
        :return: None
        """

        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit saves training data for KNN without training phase (lazy learning)
        :param X: training samples, shape (n_samples, n_features)
        :param y: target labels, shape (n_samples,)
        :return: None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict method
        For each test sample:
        - compute Euclidean distances to all training samples
        - select k nearest neighbors
        - take the mean of their target values as the prediction
        :param X_test: test samples, shape (n_samples, n_features)
        :return: predicted values as np.ndarray
        """
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

print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_my))
print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my))
print("\n")
print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_sk))
print("Sklearn KNNRegressor R2:", r2_score(yr_test, yr_pred_sk))
