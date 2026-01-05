import numpy as np
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class CustomKernelRegression:
    """
    Kernel Regression (Nadaraya-Watson estimator) for regression tasks.

    Supported kernels:
    - 'rbf' : Gaussian kernel
    - 'linear' : Linear kernel
    """

    def __init__(self, kernel: str = "rbf", bandwidth: float = 1.0):
        """
        Parameters
            kernel : str
                Kernel type ('rbf' or 'linear').
            bandwidth : float
                Bandwidth parameter for RBF kernel.
        """

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None

    def _rbf_kernel(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Gaussian (RBF) kernel between all points in X and a single point x.
        """

        return np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * self.bandwidth ** 2))

    def _linear_kernel(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Linear kernel between all points in X and a single point x.
        """

        return X.dot(x)

    def _compute_weights(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights between training data and a single point x.
        """

        if self.kernel == "rbf":
            return self._rbf_kernel(X, x)
        elif self.kernel == "linear":
            return self._linear_kernel(X, x)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Store training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression values for new data.
        """

        y_pred = []
        for x in X:
            weights = self._compute_weights(self.X_train, x)
            y_hat = np.sum(weights * self.y_train) / (np.sum(weights) + 1e-9)
            y_pred.append(y_hat)
        return np.array(y_pred)

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

custom_kr = CustomKernelRegression(kernel='rbf', bandwidth=2.0)
custom_kr.fit(X_train, y_train)
y_pred_custom = custom_kr.predict(X_test)
mse_custom = mean_squared_error(y_test, y_pred_custom)
r2_custom = r2_score(y_test, y_pred_custom)

sklearn_kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=1/(2*2.0**2))
sklearn_kr.fit(X_train, y_train)
y_pred_sklearn = sklearn_kr.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print(f"MSE (Custom RBF Kernel)    = {mse_custom:.3f}")
print(f"R2 (Custom RBF Kernel)    = {r2_custom:.3f}")
print("=" * 40)
print(f"MSE (Sklearn KernelRidge)  = {mse_sklearn:.3f}")
print(f"R2 (Sklearn RBF Kernel)    = {r2_sklearn:.3f}")