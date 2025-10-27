import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


class LinearSVR:
    def __init__(self,
                 C: float = 1.0,
                 eps: float = 0.1,
                 lr: float = 0.001,
                 n_iters: int = 1000,
                 lambda_param: float = 0.001):
        self.C = C
        self.eps = eps
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                y_pred = np.dot(x_i, self.w) + self.b
                error = y_pred - y[i]

                if error > self.eps:
                    grad_w = 2 * self.lambda_param * self.w + self.C * x_i
                    grad_b = self.C

                elif error < -self.eps:
                    grad_w = 2 * self.lambda_param * self.w - self.C * x_i
                    grad_b = -self.C

                else:
                    grad_w = 2 * self.lambda_param * self.w
                    grad_b = 0

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X):
        return np.dot(X, self.w) + self.b

X, y = datasets.make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_svr = LinearSVR(C=1.0, eps=0.1, lr=0.001, n_iters=5000, lambda_param=0.01)
custom_svr.fit(X_train, y_train)
y_pred_custom = custom_svr.predict(X_test)
mse_custom = mean_squared_error(y_test, y_pred_custom)
print("Custom Linear SVR MSE:", mse_custom)

sklearn_svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
sklearn_svr.fit(X_train, y_train)
y_pred_sklearn = sklearn_svr.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print("Sklearn Linear SVR MSE:", mse_sklearn)
