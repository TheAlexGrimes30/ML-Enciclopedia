import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_pdf(self, class_idx: int, x):
        mean_ = self.mean[class_idx]
        var_ = self.var[class_idx]
        numerator = np.exp(-((x - mean_) ** 2) / (2 * var_))
        denominator = np.sqrt(2 * np.pi * var_)
        return numerator / denominator

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self._gaussian_pdf(c, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

X, y = make_classification(
    n_samples=300, n_features=4, n_informative=4,
    n_redundant=0, n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_gbb = GaussianNaiveBayes()
my_gbb.fit(X_train, y_train)
my_y_pred = my_gbb.predict(X_test)

sk_gnb = GaussianNB()
sk_gnb.fit(X_train, y_train)
sk_y_pred = sk_gnb.predict(X_test)

print("Custom GaussianNB accuracy:", accuracy_score(y_test, my_y_pred))
print("Sklearn GaussianNB accuracy:", accuracy_score(y_test, sk_y_pred))

