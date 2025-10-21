import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class LinearRegressionCustom:
    def __init__(self,
                 lr: float = 0.1,
                 n_iters: int = 1000,
                 regularization: str = None,
                 alpha: float = 0.1,
                 l1_ratio: float = 0.5
                 ):
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b
            dw = -(1 / m) * np.dot(X.T, (y - y_pred))
            db = -(1 / m) * np.sum(y - y_pred)

            if self.regularization == "l2":
                dw += (self.alpha / m) * self.w
            elif self.regularization == "l1":
                dw += (self.alpha / m) * np.sign(self.w)
            elif self.regularization == "elasticnet":
                dw += (self.alpha / m) * (
                    self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * self.w
                )

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regularizations = [None, 'l1', 'l2', 'elasticnet']

print("=== Custom Linear Regression Models ===")
for reg in regularizations:
    model = LinearRegressionCustom(lr=0.01, n_iters=1000, regularization=reg, alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Regularization = {reg or 'None'} | R2 = {r2_score(y_test, y_pred):.3f}")

print("\n=== Sklearn Models ===")
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name:15} | R2 = {r2_score(y_test, y_pred):.3f}")
