import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


class BernoulliNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        self.class_log_prior = {}
        self.feature_log_prob = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_log_prior[c] = np.log(X_c.shape[0] / n_samples)
            feature_prob = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)
            self.feature_log_prob[c] = np.log(feature_prob)
            self.feature_log_prob[f"{c}_neg"] = np.log(1 - feature_prob)

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_likelihood = np.sum(
                    x * self.feature_log_prob[c] + (1 - x) * self.feature_log_prob[f"{c}_neg"]
                )
                posterior = self.class_log_prior[c] + log_likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

X, y = make_classification(
    n_samples=300, n_features=4, n_informative=4,
    n_redundant=0, n_classes=2, random_state=42
)

X = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_mnb = BernoulliNaiveBayes()
my_mnb.fit(X_train, y_train)
my_y_pred = my_mnb.predict(X_test)

sk_bnb = BernoulliNB()
sk_bnb.fit(X_train, y_train)
sk_y_pred = sk_bnb.predict(X_test)

print("Custom BernoulliNB accuracy:", accuracy_score(y_test, my_y_pred))
print("Sklearn BernoulliNB accuracy:", accuracy_score(y_test, sk_y_pred))
