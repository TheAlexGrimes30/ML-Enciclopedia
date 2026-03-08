import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    """
    KNN Classifier class
    methods: constructor, fit, predict
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", p: int = 2):
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
         - select k nearest samples
         - assign the most common label among them
        :param X_test: test samples, shape (n_samples, n_features)
        :return: predicted labels as np.ndarray
        """

        predictions = []
        for x in X_test:
            distances = self._distance(x)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)


X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_knn = KNNClassifier(k=3)
my_knn.fit(X_train, y_train)
y_pred_my = my_knn.predict(X_test)

my_knn_manhattan = KNNClassifier(k=3, metric="manhattan")
my_knn_manhattan.fit(X_train, y_train)
y_pred_my_manhattan = my_knn_manhattan.predict(X_test)

my_knn_minkowski = KNNClassifier(k=3, metric="minkowski")
my_knn_minkowski.fit(X_train, y_train)
y_pred_my_minkowski = my_knn_minkowski.predict(X_test)

my_knn_cosine = KNNClassifier(k=3, metric="cosine")
my_knn_cosine.fit(X_train, y_train)
y_pred_my_cosine = my_knn_cosine.predict(X_test)

sk_knn = KNeighborsClassifier(n_neighbors=3)
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)

print("Euclidean")
print("Custom KNNClassifier accuracy:", accuracy_score(y_test, y_pred_my))
print("Custom KNNClassifier precision:", precision_score(y_test, y_pred_my))
print("Custom KNNClassifier recall:", recall_score(y_test, y_pred_my))
print("Custom KNNClassifier f1 score:", f1_score(y_test, y_pred_my))
print("Custom KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_my))
print("\n")
print("Manhattan")
print("Custom KNNClassifier accuracy:", accuracy_score(y_test, y_pred_my_manhattan))
print("Custom KNNClassifier precision:", precision_score(y_test, y_pred_my_manhattan))
print("Custom KNNClassifier recall:", recall_score(y_test, y_pred_my_manhattan))
print("Custom KNNClassifier f1 score:", f1_score(y_test, y_pred_my_manhattan))
print("Custom KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_my_manhattan))
print("\n")
print("Minkowski")
print("Custom KNNClassifier accuracy:", accuracy_score(y_test, y_pred_my_minkowski))
print("Custom KNNClassifier precision:", precision_score(y_test, y_pred_my_minkowski))
print("Custom KNNClassifier recall:", recall_score(y_test, y_pred_my_minkowski))
print("Custom KNNClassifier f1 score:", f1_score(y_test, y_pred_my_minkowski))
print("Custom KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_my_minkowski))
print("\n")
print("Cosine")
print("Custom KNNClassifier accuracy:", accuracy_score(y_test, y_pred_my_cosine))
print("Custom KNNClassifier precision:", precision_score(y_test, y_pred_my_cosine))
print("Custom KNNClassifier recall:", recall_score(y_test, y_pred_my_cosine))
print("Custom KNNClassifier f1 score:", f1_score(y_test, y_pred_my_cosine))
print("Custom KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_my_cosine))
print("\n")
print("Sklearn KNNClassifier accuracy:", accuracy_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier precision:", precision_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier recall:", recall_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier f1 score:", f1_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_sk))