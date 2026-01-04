import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X, _ = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class CustomPCA:
    """
    Principal Component Analysis (PCA)
    based on eigen decomposition of the covariance matrix.
    """

    def __init__(self, n_components: int):
        """
        Parameters
            n_components : int
                Number of principal components to retain.
        """

        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "CustomPCA":
        """
        Fit the PCA model by computing principal components.

        Parameters
            X : np.ndarray
                Scaled data matrix of shape (n_samples, n_features).
        """

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.components_ = eigenvectors[:, idx][:, :self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.

        Parameters
            X : np.ndarray
                Data matrix of shape (n_samples, n_features).

        Returns
            np.ndarray
                Transformed data of shape (n_samples, n_components).
        """

        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal components.

        Parameters
            Z : np.ndarray
                Low-dimensional representation.

        Returns
            np.ndarray
                Reconstructed data in original feature space.
        """

        return Z @ self.components_.T + self.mean_

    def explained_variance_ratio(self) -> np.ndarray:
        """
        Compute explained variance ratio.

        Returns
            np.ndarray
                Explained variance ratio for each component.
        """

        return self.eigenvalues_[:self.n_components] / np.sum(self.eigenvalues_)

sklearn_pca = PCA(n_components=5)
sklearn_pca.fit(X_scaled)

custom_pca = CustomPCA(n_components=5)
custom_pca.fit(X_scaled)

Z_custom = custom_pca.transform(X_scaled)
Z_sklearn = sklearn_pca.transform(X_scaled)

X_rec_custom = custom_pca.inverse_transform(Z_custom)
X_rec_sklearn = sklearn_pca.inverse_transform(Z_sklearn)

reconstruction_error_custom = mean_squared_error(X_scaled, X_rec_custom)
reconstruction_error_sklearn = mean_squared_error(X_scaled, X_rec_sklearn)

explained_var_custom = np.sum(custom_pca.explained_variance_ratio())
explained_var_sklearn = np.sum(sklearn_pca.explained_variance_ratio_)

print("Custom PCA")
print(f"Explained variance ratio: {explained_var_custom:.4f}")
print(f"Reconstruction MSE:       {reconstruction_error_custom:.6f}")
print("-" * 40)

print("Sklearn PCA")
print(f"Explained variance ratio: {explained_var_sklearn:.4f}")
print(f"Reconstruction MSE:       {reconstruction_error_sklearn:.6f}")
