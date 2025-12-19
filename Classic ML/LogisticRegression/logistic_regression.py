import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LogisticRegressionCustom:
    """
    Logistic Regression with support for
    different optimization solvers and regularization techniques.

    Supported solvers:
    - Gradient Descent (gd)
    - Stochastic Average Gradient (sag)
    - Newton-CG (newton-cg)

    Supported regularization:
    - L1
    - L2
    - ElasticNet
    """
    def __init__(self, lr: float = 0.01,
                 n_iters: int = 1000,
                 regularization: str = None,
                 alpha: float = 0.1,
                 l1_ratio: float = 0.5,
                 solver: str = "gd"
                 ):
        """
        Constructor of Logistic Regression model.

        Parameters
        ----------
        lr : float
            Learning rate for optimization.
        n_iters : int
            Number of training iterations.
        regularization : str or None
            Type of regularization ('l1', 'l2', 'elasticnet' or None).
        alpha : float
            Regularization strength.
        l1_ratio : float
            Ratio between L1 and L2 penalties for ElasticNet.
        solver : str
            Optimization algorithm ('gd', 'sag', 'newton-cg').
        """

        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.w = None
        self.b = 0
        self.solver = solver

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of features and weights.

        Returns
        -------
        np.ndarray
            Sigmoid-transformed values.
        """

        return 1 / (1 + np.exp(-z))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic loss (binary cross-entropy).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).

        Returns
        -------
        float
            Mean logistic loss value.
        """

        y_pred = self._sigmoid(np.dot(X, self.w) + self.b)
        return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

    def _add_regularization(self, dw: np.ndarray) -> np.ndarray:
        """
        Apply regularization term to the gradient.

        Parameters
        ----------
        dw : np.ndarray
            Gradient of the loss with respect to weights.

        Returns
        -------
        np.ndarray
            Regularized gradient.
        """

        if self.regularization == "l2":
            dw += (self.alpha) * self.w

        elif self.regularization == "l1":
            dw += (self.alpha) * np.sign(self.w)

        elif self.regularization == "elasticnet":
            dw += (self.alpha) * (self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * self.w)

        return dw

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).

        Returns
        -------
        None
        """

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        if self.solver == "gd":
            for _ in range(self.n_iters):
                linear = np.dot(X, self.w) + self.b
                y_pred = self._sigmoid(linear)
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                dw = self._add_regularization(dw)
                self.w -= self.lr * dw
                self.b -= self.lr * db

        elif self.solver == "sag":
            avg_grad = np.zeros_like(self.w)
            for _ in range(self.n_iters):
                i = np.random.randint(0, n_samples)
                xi, yi = X[i], y[i]
                y_pred = self._sigmoid(np.dot(xi, self.w) + self.b)
                grad = xi * (y_pred - yi)
                avg_grad = 0.9 * avg_grad + 0.1 * grad
                dw = avg_grad
                dw = self._add_regularization(dw)
                self.w -= self.lr * dw
                self.b -= self.lr * (y_pred - yi)

        elif self.solver == "newton-cg":
            for _ in range(self.n_iters):
                y_pred = self._sigmoid(np.dot(X, self.w) + self.b)
                grad = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                H = (1 / n_samples) * np.dot(X.T * (y_pred * (1 - y_pred)), X)
                dw = np.linalg.pinv(H).dot(grad)
                dw = self._add_regularization(dw)
                self.w -= dw
                self.b -= np.mean(y_pred - y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted probabilities for class 1.
        """

        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).
        """

        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

X, y = make_classification(n_samples=300, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

solvers = ["gd", "sag", "newton-cg"]
for s in solvers:
    model = LogisticRegressionCustom(lr=0.1, n_iters=500, solver=s, regularization='l2', alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Custom Solver={s:10s}  Accuracy={accuracy_score(y_test, y_pred):.3f}")

sk_model = LogisticRegression(solver='lbfgs', penalty='l2', C=10, max_iter=1000)
sk_model.fit(X_train, y_train)
y_pred_sk = sk_model.predict(X_test)
print(f"Sklearn LogisticRegression (lbfgs) Accuracy={accuracy_score(y_test, y_pred_sk):.3f}")