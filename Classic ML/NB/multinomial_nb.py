import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_count = {c: np.sum(y == c) for c in self.classes}
        self.feature_count = {c: np.sum(X[y == c], 0) for c in self.classes}
        self.feature_log_prob = {}
        self.class_log_prior = {}

        for c in self.classes:
            smoothed = (self.feature_count[c] + 1) / (np.sum(self.feature_count[c]) + X.shape[1])
            self.feature_log_prob[c] = np.log(smoothed)
            self.class_log_prior[c] = np.log(self.class_count[c] / X.shape[0])

    def predict(self, X):
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

sk_gnb = MultinomialNB()
sk_gnb.fit(X_train, y_train)
sk_y_pred = sk_gnb.predict(X_test)

print("Custom MultinomialNB accuracy:", accuracy_score(y_test, my_y_pred))
print("Sklearn MultinomialNB accuracy:", accuracy_score(y_test, sk_y_pred))