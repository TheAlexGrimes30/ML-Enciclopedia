import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressorCustom:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_thresh = 0, 0
        current_mse = self._mse(y)
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])

            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                p_left = len(y_left) / n_samples
                gain = current_mse - (p_left * self._mse(y_left) + (1 - p_left) * self._mse(y_right))

                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, t

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth: int = 0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))

        feat, thresh, gain = self._best_split(X, y)
        if gain == 0:
            return Node(value=np.mean(y))

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node: Node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

class CustomGradientBoostingRegressor:
    def __init__(self, base_estimator=DecisionTreeRegressorCustom,
                 n_estimators: int = 10, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.models = []
        self.gammas = []
        self.F0 = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        n = y.shape[0]

        self.F0 = np.mean(y)
        Fm = np.full(n, self.F0, dtype=float)

        self.models = []
        self.gammas = []

        for m in range(self.n_estimators):
            residuals = y - Fm

            tree = self.base_estimator(max_depth=self.max_depth)
            tree.fit(X, residuals)
            h = tree.predict(X)

            if np.allclose(h, 0):
                break

            denom = np.dot(h, h)
            if denom == 0:
                gamma = 0.0
            else:
                gamma = np.dot(residuals, h) / (denom + 1e-12)

            if np.abs(gamma) < 1e-12:
                break

            Fm = Fm + self.learning_rate * gamma * h

            self.models.append(tree)
            self.gammas.append(gamma)

    def predict(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        Fm = np.full(n, self.F0, dtype=float)
        for tree, gamma in zip(self.models, self.gammas):
            Fm += self.learning_rate * gamma * tree.predict(X)
        return Fm

if __name__ == "__main__":
    X, y = make_regression(n_samples=500, n_features=8, n_informative=6, noise=15, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gb_custom = CustomGradientBoostingRegressor(
        base_estimator=DecisionTreeRegressorCustom,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3
    )
    gb_custom.fit(X_train, y_train)
    y_pred_custom = gb_custom.predict(X_test)

    gb_sklearn = GradientBoostingRegressor(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_sklearn.fit(X_train, y_train)
    y_pred_sklearn = gb_sklearn.predict(X_test)

    print("Custom GB MSE:", mean_squared_error(y_test, y_pred_custom))
    print("Custom GB R2: ", r2_score(y_test, y_pred_custom))
    print("---")
    print("Sklearn GB MSE:", mean_squared_error(y_test, y_pred_sklearn))
    print("Sklearn GB R2: ", r2_score(y_test, y_pred_sklearn))
