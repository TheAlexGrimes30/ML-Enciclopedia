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
    def __init__(self, k: int = 3) -> None:
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
         - select k nearest samples
         - assign the most common label among them
        :param X_test: test samples, shape (n_samples, n_features)
        :return: predicted labels as np.ndarray
        """
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, 1))
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

sk_knn = KNeighborsClassifier(n_neighbors=5)
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)

print("Custom KNNClassifier accuracy:", accuracy_score(y_test, y_pred_my))
print("Custom KNNClassifier precision:", precision_score(y_test, y_pred_my))
print("Custom KNNClassifier recall:", recall_score(y_test, y_pred_my))
print("Custom KNNClassifier f1 score:", f1_score(y_test, y_pred_my))
print("Custom KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_my))
print("\n")
print("Sklearn KNNClassifier accuracy:", accuracy_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier precision:", precision_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier recall:", recall_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier f1 score:", f1_score(y_test, y_pred_sk))
print("Sklearn KNNClassifier roc auc score:", roc_auc_score(y_test, y_pred_sk))