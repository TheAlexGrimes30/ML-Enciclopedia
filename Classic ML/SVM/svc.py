import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class LinearSVC:
    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iters: int = 5000, lambda_param: float = 0.001):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - self.C * y_[idx] * x_i)
                    self.b -= self.lr * self.C * y_[idx]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))



X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2,
                                    n_redundant=0, n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_svm = LinearSVC(C=1.0, lr=0.001, n_iters=1000, lambda_param=0.01)
custom_svm.fit(X_train, y_train)
y_pred_custom = custom_svm.predict(X_test)
acc_custom = accuracy_score(y_test, y_pred_custom)
print("Custom Linear SVM Accuracy:", acc_custom)

sklearn_svm = SVC(kernel='linear', C=1.0)
sklearn_svm.fit(X_train, y_train)
y_pred_sklearn = sklearn_svm.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Sklearn Linear SVM Accuracy:", acc_sklearn)
