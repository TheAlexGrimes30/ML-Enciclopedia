import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

accuracy_custom = (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy_custom = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
precision_custom = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_custom = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_custom = 2 * precision_custom * recall_custom / (precision_custom + recall_custom)

accuracy_skl = accuracy_score(y_test, y_pred)
balanced_accuracy_skl = balanced_accuracy_score(y_test, y_pred)
precision_skl = precision_score(y_test, y_pred)
recall_skl = recall_score(y_test, y_pred)
f1_skl = f1_score(y_test, y_pred)
roc_auc_skl = roc_auc_score(y_test, y_prob)

print("Метрики (кастомные):")
print(f"Accuracy: {accuracy_custom:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_custom:.3f}")
print(f"Precision: {precision_custom:.3f}")
print(f"Recall: {recall_custom:.3f}")
print(f"F1 Score: {f1_custom:.3f}")

print("Метрики (sklearn):")
print(f"Accuracy: {accuracy_skl:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_skl:.3f}")
print(f"Precision: {precision_skl:.3f}")
print(f"Recall: {recall_skl:.3f}")
print(f"F1 Score: {f1_skl:.3f}")
print(f"ROC AUC: {roc_auc_skl:.3f}")