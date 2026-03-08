from collections import Counter

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from annoy import AnnoyIndex
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

k = 5
dim = X_train.shape[1]

annoy_index = AnnoyIndex(dim, "euclidean")

for i, vector in enumerate(X_train):
    annoy_index.add_item(i, vector)

annoy_index.build(10)

predictions_annoy = []

for x in X_test:
    neighbors = annoy_index.get_nns_by_vector(x, k)
    neighbor_labels = y_train[neighbors]
    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    predictions_annoy.append(prediction)

acc_annoy = accuracy_score(y_test, predictions_annoy)
print("Annoy Accuracy:", acc_annoy)