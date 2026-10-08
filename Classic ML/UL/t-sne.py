import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness, TSNE
from sklearn.preprocessing import StandardScaler


class TSNECustom:
    def __init__(
            self,
            n_components: int = 2,
            perplexity: float = 30.0,
            learning_rate: float = 200.0,
            n_iter: int = 1000,
            early_exaggeration: float = 12.0,
            early_exaggeration_iter: int = 250,
            momentum_start: float = 0.5,
            momentum_end: float = 0.8,
            momentum_switch_iter: int = 250,
            min_gain: float = 0.01,
            tol: float = 1e-5,
            binary_search_iters: int = 60,
            random_state: int = 42,
            verbose: bool = True
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.momentum_switch_iter = momentum_switch_iter
        self.min_gain = min_gain
        self.tol = tol
        self.binary_search_iters = binary_search_iters
        self.random_state = random_state
        self.verbose = verbose

        self.embedding_ = None
        self.kl_history_ = []

    @staticmethod
    def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
        squared_norms = np.sum(X * X, axis=1, keepdims=True)
        distances = squared_norms + squared_norms.T - 2.0 * (X @ X.T)

        return np.maximum(distances, 0.0)

    @staticmethod
    def _conditional_probabilities_for_beta(
            distances_i: np.ndarray,
            beta: float,
            self_index: int
    ) -> tuple[np.ndarray, float]:
        probabilities = np.exp(-distances_i * beta)
        probabilities[self_index] = 0.0

        sum_probabilities = np.sum(probabilities)

        if sum_probabilities <= 1e-300:
            probabilities = np.zeros_like(distances_i)
            mask = np.ones_like(distances_i, dtype=bool)
            mask[self_index] = False
            probabilities[mask] = 1.0 / np.sum(mask)
        else:
            probabilities /= sum_probabilities

        mask = probabilities > 0.0
        entropy = -np.sum(probabilities[mask] * np.log(probabilities[mask] + 1e-300))

        return probabilities, entropy

    def _joint_probabilities(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]

        if self.perplexity >= n_samples:
            raise ValueError("perplexity must be smaller than number of samples.")

        distances = self._pairwise_squared_distances(X)

        conditional = np.zeros(
            (n_samples, n_samples),
            dtype=np.float64,
        )

        target_entropy = np.log(self.perplexity)

        for i in range(n_samples):
            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf

            probabilities_i = None

            for _ in range(self.binary_search_iters):
                probabilities_i, entropy = (
                    self._conditional_probabilities_for_beta(
                        distances_i=distances[i],
                        beta=beta,
                        self_index=i,
                    )
                )

                entropy_diff = entropy - target_entropy

                if abs(entropy_diff) < self.tol:
                    break

                if entropy_diff > 0.0:
                    beta_min = beta

                    if np.isinf(beta_max):
                        beta *= 2.0
                    else:
                        beta = 0.5 * (beta + beta_max)
                else:
                    beta_max = beta

                    if np.isinf(beta_min):
                        beta *= 0.5
                    else:
                        beta = 0.5 * (beta + beta_min)

            conditional[i] = probabilities_i

        P = (conditional + conditional.T) / (2.0 * n_samples)

        P = np.maximum(P, 1e-12)
        np.fill_diagonal(P, 0.0)
        P /= np.sum(P)

        return P

    @staticmethod
    def _kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        mask = P > 0.0

        return float(np.sum(P[mask] * np.log((P[mask] + 1e-12) / (Q[mask] + 1e-12))))

    @staticmethod
    def _low_dimensional_probabilities(
            Y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        distances = TSNECustom._pairwise_squared_distances(Y)

        num = 1.0 / (1.0 + distances)
        np.fill_diagonal(num, 0.0)

        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        np.fill_diagonal(Q, 0.0)
        Q /= np.sum(Q)

        return Q, num

    @staticmethod
    def _gradient(
            Y: np.ndarray,
            P: np.ndarray,
            Q: np.ndarray,
            num: np.ndarray,
    ) -> np.ndarray:

        pq = (P - Q) * num
        row_sums = np.sum(pq, axis=1)
        gradient = 4.0 * (row_sums[:, None] * Y - pq @ Y)

        return gradient

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

        n_samples = X.shape[0]

        if n_samples < 2:
            raise ValueError("At least 2 samples are required.")

        rng = np.random.default_rng(self.random_state)
        X = X - X.mean(axis=0, keepdims=True)
        P = self._joint_probabilities(X)

        Y = rng.normal(
            loc=0.0,
            scale=1e-4,
            size=(n_samples, self.n_components),
        )

        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)

        self.kl_history_ = []

        for iteration in range(self.n_iter):
            if iteration < self.early_exaggeration_iter:
                P_used = P * self.early_exaggeration
            else:
                P_used = P

            Q, num = self._low_dimensional_probabilities(Y)

            gradient = self._gradient(
                Y=Y,
                P=P_used,
                Q=Q,
                num=num,
            )

            momentum = (
                self.momentum_start
                if iteration < self.momentum_switch_iter
                else self.momentum_end
            )

            sign_changed = np.sign(gradient) != np.sign(velocity)
            gains = np.where(sign_changed, gains + 0.2, gains * 0.8)

            gains = np.maximum(gains, self.min_gain)
            velocity = momentum * velocity - self.learning_rate * gains * gradient

            Y += velocity

            Y -= Y.mean(axis=0, keepdims=True)
            Q_eval, _ = self._low_dimensional_probabilities(Y)
            kl = self._kl_divergence(P, Q_eval)
            self.kl_history_.append(kl)

            if self.verbose and (
                    iteration == 0
                    or (iteration + 1) % 100 == 0
                    or iteration == self.n_iter - 1
            ):
                print(
                    f"iteration={iteration + 1:04d}/{self.n_iter} "
                    f"KL={kl:.6f}"
                )

        self.embedding_ = Y
        return Y

def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(9, 7))

    for class_id in np.unique(labels):
        mask = labels == class_id

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.7,
            label=str(class_id),
        )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(title="Class", ncol=2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160)

    plt.show()


def plot_kl_history(
    kl_history: list[float],
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(kl_history) + 1),
        kl_history,
    )
    plt.xlabel("Iteration")
    plt.ylabel("KL divergence")
    plt.title("Custom t-SNE optimization")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160)

    plt.show()

def main():
    digits = load_digits()

    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    rng = np.random.default_rng(42)

    n_samples = 600
    indices = rng.choice(
        len(X),
        size=n_samples,
        replace=False,
    )

    X = X[indices]
    y = y[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Dataset shape:", X_scaled.shape)

    custom_tsne = TSNECustom(
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        early_exaggeration=12.0,
        early_exaggeration_iter=250,
        random_state=42,
        verbose=True,
    )

    custom_embedding = custom_tsne.fit_transform(X_scaled)

    sklearn_tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        max_iter=1000,
        early_exaggeration=12.0,
        init="random",
        random_state=42,
    )

    sklearn_embedding = sklearn_tsne.fit_transform(X_scaled)

    custom_trustworthiness = trustworthiness(
        X_scaled,
        custom_embedding,
        n_neighbors=10,
    )

    sklearn_trustworthiness = trustworthiness(
        X_scaled,
        sklearn_embedding,
        n_neighbors=10,
    )

    print("\n=== Comparison: Custom t-SNE vs sklearn ===")
    print(
        f"Custom final KL: "
        f"{custom_tsne.kl_history_[-1]:.6f}"
    )
    print(
        f"sklearn KL:      "
        f"{sklearn_tsne.kl_divergence_:.6f}"
    )
    print(
        f"Custom trustworthiness:  "
        f"{custom_trustworthiness:.6f}"
    )
    print(
        f"sklearn trustworthiness: "
        f"{sklearn_trustworthiness:.6f}"
    )

    plot_embedding(
        custom_embedding,
        y,
        "Custom t-SNE",
        save_path="tsne_custom.png",
    )

    plot_embedding(
        sklearn_embedding,
        y,
        "sklearn t-SNE",
        save_path="tsne_sklearn.png",
    )

    plot_kl_history(
        custom_tsne.kl_history_,
        save_path="tsne_custom_kl.png",
    )


if __name__ == "__main__":
    main()



