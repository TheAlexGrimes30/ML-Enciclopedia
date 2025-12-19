import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes Classifier
    Methods:
    - fit: learn class priors and feature probabilities
    - predict: predict class labels for new samples
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Multinomial Naive Bayes model
        :param X: training samples, shape (n_samples, n_features), discrete counts
        :param y: target labels, shape (n_samples,)
        :return: None
        """

        self.classes = np.unique(y)
        self.class_count = {c: np.sum(y == c) for c in self.classes}
        self.feature_count = {c: np.sum(X[y == c], 0) for c in self.classes}
        self.feature_log_prob = {}
        self.class_log_prior = {}

        for c in self.classes:
            smoothed = (self.feature_count[c] + 1) / (np.sum(self.feature_count[c]) + X.shape[1])
            self.feature_log_prob[c] = np.log(smoothed)
            self.class_log_prior[c] = np.log(self.class_count[c] / X.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples
        :param X: test samples, shape (n_samples, n_features)
        :return: predicted labels, shape (n_samples,)
        """

        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                posterior = self.class_log_prior[c] + np.sum(x * self.feature_log_prob[c])
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

X, y = make_classification(
    n_samples=300, n_features=4, n_informative=4,
    n_redundant=0, n_classes=2, random_state=42
)

X = X - X.min() + 1e-9

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_mnb = MultinomialNaiveBayes()
my_mnb.fit(X_train, y_train)
my_y_pred = my_mnb.predict(X_test)

sk_mnb = MultinomialNB()
sk_mnb.fit(X_train, y_train)
sk_y_pred = sk_mnb.predict(X_test)

print("Custom MultinomialNB accuracy:", accuracy_score(y_test, my_y_pred))
print("Custom MultinomialNB precision:", precision_score(y_test, my_y_pred))
print("Custom MultinomialNB recall:", recall_score(y_test, my_y_pred))
print("Custom MultinomialNB f1:", f1_score(y_test, my_y_pred))
print("Custom MultinomialNB roc-auc:", roc_auc_score(y_test, my_y_pred))
print()
print("Sklearn MultinomialNB accuracy:", accuracy_score(y_test, sk_y_pred))
print("Sklearn MultinomialNB precision:", precision_score(y_test, my_y_pred))
print("Sklearn MultinomialNB recall:", recall_score(y_test, my_y_pred))
print("Sklearn MultinomialNB f1:", f1_score(y_test, my_y_pred))
print("Sklearn MultinomialNB roc-auc:", roc_auc_score(y_test, my_y_pred))