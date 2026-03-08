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

    def __init__(self, k: int = 2, metric: str = "euclidean", p: int = 2):
        """
        Constructor
        :param k: number of neighbours
        :param metric: distance metric (euclidean, manhattan, minkowski, cosine)
        :param p: parameter for minkowski distance
        """

        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit saves training data for KNN without training phase (lazy learning)
        :param X: training samples, shape (n_samples, n_features)
        :param y: target labels, shape (n_samples,)
        :return: None
        """

        self.X_train = X
        self.y_train = y

    def _distance(self, x: np.ndarray) -> np.ndarray:

        """
        Calculate distance between one test sample x and all training samples.

        Depending on the selected metric, this function computes:
        - Euclidean distance
        - Manhattan distance
        - Minkowski distance
        - Cosine distance

        :param x: one test sample (1D array of features)
        :return: array of distances between x and each training sample
        """

        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        elif self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)

        elif self.metric == "minkowski":
            return np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)

        elif self.metric == "cosine":
            dot = np.dot(self.X_train, x)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_x = np.linalg.norm(x)
            cosine_similarity_custom = dot / (norm_train * norm_x + 1e-10)
            return 1 - cosine_similarity_custom

        else:
            raise ValueError("Unsupported metric")

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
            distances = self._distance(x)
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

my_knn_reg_manhattan = KNNRegressor(k=5, metric="manhattan")
my_knn_reg_manhattan.fit(Xr_train, yr_train)
yr_pred_my_manhattan = my_knn_reg_manhattan.predict(Xr_test)

my_knn_reg_minkowski = KNNRegressor(k=5, metric="minkowski")
my_knn_reg_minkowski.fit(Xr_train, yr_train)
yr_pred_my_minkowski = my_knn_reg_minkowski.predict(Xr_test)

my_knn_reg_cosine = KNNRegressor(k=5, metric="cosine")
my_knn_reg_cosine.fit(Xr_train, yr_train)
yr_pred_my_cosine = my_knn_reg_cosine.predict(Xr_test)

sk_knn_reg = KNeighborsRegressor(n_neighbors=5)
sk_knn_reg.fit(Xr_train, yr_train)
yr_pred_sk = sk_knn_reg.predict(Xr_test)

print("Custom KNNRegressor Euclidean")
print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_my))
print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my))
print("\n")
print("Custom KNNRegressor Manhattan")
print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_my_manhattan))
print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my_manhattan))
print("\n")
print("Custom KNNRegressor Minkowski")
print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_my_minkowski))
print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my_minkowski))
print("\n")
print("Custom KNNRegressor Cosine")
print("Custom KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_my_cosine))
print("Custom KNNRegressor R2:", r2_score(yr_test, yr_pred_my_cosine))
print("\n")
print("Sklearn KNNRegressor MSE:", mean_squared_error(yr_test, yr_pred_sk))
print("Sklearn KNNRegressor R2:", r2_score(yr_test, yr_pred_sk))
