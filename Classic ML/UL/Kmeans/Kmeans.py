import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


class KMeansScratch:
    def __init__(self, n_clusters: int =3, max_iters: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                                      for k in range(self.n_clusters)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def inertia_(self, X):
        return np.sum((X - self.centroids[self.labels_]) ** 2)

X, y_true = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)

kmeans_scratch = KMeansScratch(n_clusters=3, random_state=42)
kmeans_scratch.fit(X)
scratch_inertia = kmeans_scratch.inertia_(X)
scratch_ari = adjusted_rand_score(y_true, kmeans_scratch.labels_)
scratch_silhouette = silhouette_score(X, kmeans_scratch.labels_)

kmeans_sklearn = SklearnKMeans(n_clusters=3, random_state=42)
kmeans_sklearn.fit(X)
sklearn_inertia = kmeans_sklearn.inertia_
sklearn_ari = adjusted_rand_score(y_true, kmeans_sklearn.labels_)
sklearn_silhouette = silhouette_score(X, kmeans_sklearn.labels_)

print("=== Сравнение K-Means Scratch и sklearn ===")
print(f"Scratch Inertia: {scratch_inertia:.2f}")
print(f"Sklearn Inertia: {sklearn_inertia:.2f}")
print(f"Scratch ARI: {scratch_ari:.4f}")
print(f"Sklearn ARI: {sklearn_ari:.4f}")
print(f"Scratch Silhouette: {scratch_silhouette:.4f}")
print(f"Sklearn Silhouette: {sklearn_silhouette:.4f}")