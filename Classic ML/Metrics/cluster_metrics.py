from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(
    n_samples=500,
    n_features=5,
    centers=3,
    cluster_std=1.5,
    random_state=42,
    shuffle=False
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
y_pred = kmeans.labels_
centroids = kmeans.cluster_centers_

sklearn_metrics = {
    'Silhouette': silhouette_score(X_scaled, y_pred),
    'Adjusted Rand Score': adjusted_rand_score(y_true, y_pred),
    'Davies-Bouldin': davies_bouldin_score(X_scaled, y_pred),
    'Calinski-Harabasz': calinski_harabasz_score(X_scaled, y_pred),
    'Inertia': kmeans.inertia_
}

print("\n" + "=" * 60)
print("МЕТРИКИ (Sklearn РАСЧЕТЫ):")
print("=" * 60)
for metric, value in sklearn_metrics.items():
    print(f"{metric:<2}: {value:.4f}")
