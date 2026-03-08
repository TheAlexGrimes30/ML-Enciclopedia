from collections import Counter

import faiss
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(X_train)
distances, indices = faiss_index.search(X_test, k)
predictions_faiss = []

for neighbors in indices:
    neighbor_labels = y_train[neighbors]
    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    predictions_faiss.append(prediction)

acc_faiss = accuracy_score(y_test, predictions_faiss)
precision_faiss = precision_score(y_test, predictions_faiss)
recall_faiss = recall_score(y_test, predictions_faiss)
f1_faiss = f1_score(y_test, predictions_faiss)

print("Faiss Accuracy:", acc_faiss)
print("Faiss Precision:", precision_faiss)
print("Faiss Recall:", recall_faiss)
print("Faiss F1:", f1_faiss)
