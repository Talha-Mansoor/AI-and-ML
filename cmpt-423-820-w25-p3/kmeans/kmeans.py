import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import click
import numpy as np


@dataclass
class KMeansConfig:
    """This dataclass stores the KMeans clustering algorithm hyper-parameter.

    Parameter:
    * k: Number of centroids
    * max_iterations: Maximum number of improvement iterations.
    * epsilon: Distortion improvement threshold. If one iteration does not
        improve distortion by at least epsilon, then the algorithm stops early.
    """

    k: int
    max_iterations: int = 100
    epsilon: float = 1e-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
        }


@dataclass
class KMeansResult:
    """This dataclass stored the clustering result of the KMeans clustering
    algorithm.

    The matrix `mu` has shape `[k, d]` for `k` clusters of dimension `d`. The
    `membership` matrix has shape `[n, k]` for `n` data points (images) and
    `k` clusters. This matrix can be stored using datatype float32.

    """

    config: KMeansConfig
    distortion: float
    mu: np.ndarray
    membership: np.ndarray

    def to_file(self, filename: str):
        kmeans_res_dict = {
            "config": self.config.to_dict(),
            "distortion": float(self.distortion),
            "mu": self.mu.tolist(),
            "membership": self.membership.tolist(),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(kmeans_res_dict, f)

    @staticmethod
    def from_file(filename: str) -> "KMeansResult":
        with open(filename, "r", encoding="utf-8") as f:
            res_dict = json.load(f)
        mu = np.array(res_dict["mu"], dtype=np.float32)
        membership = np.array(res_dict["membership"], dtype=np.float32)
        return KMeansResult(
            config=KMeansConfig(**res_dict["config"]),
            distortion=res_dict["distortion"],
            mu=mu,
            membership=membership,
        )


@dataclass
class MNIST7K:
    """Dataclass for loading the MNIST dataset used in this project."""

    imgs: np.ndarray
    labels: np.ndarray

    @staticmethod
    def from_file() -> "MNIST7K":
        imgs_256 = np.load("data/mnist_imgs.npy").reshape(-1, 28, 28)
        return MNIST7K(
            imgs=imgs_256.astype(np.float32) / 255.0,
            labels=np.load("data/mnist_labels.npy"),
        )


def compute_distortion(
    x: np.ndarray,
    mu: np.ndarray,
    membership: np.ndarray,
) -> float:
    """Compute the distortion J (the loss objective that K-Means
    clustering optimizes.)

    Args:
        x (np.ndarray): Dataset matrix of shape `[n, d]`.
        mu (np.ndarray): Centroid matrix of shape `[k, d]`.
        membership (np.ndarray): Cluster membership bit matrix of shape
            `[n, k]`.

    Returns:
        float: Distortion value (value of the J loss objective).
    """
    # insert your code here
    j = np.sum(np.pow((x - np.dot(membership, mu)), 2))
    return j


class KMeans:
    def __init__(self, config: KMeansConfig):
        """K-Means clustering implementation.

        Args:
            config (KMeansConfig): KMeans clustering config.
        """
        self.config = config

    def init_centroids(self, x: np.ndarray) -> np.ndarray:
        """Initialize centroids by randomly picking points from the dataset.
        Start points can be picked uniformly at random.

        Args:
            x (np.ndarray): Dataset array `[n, d]`.

        Returns:
            np.ndarray: Centroid array of shape `[k, d]`.
        """
        choices = np.random.choice(
            x.shape[0], size=self.config.k, replace=False
        )
        centroids = x[choices]
        return centroids

    def fit(
        self,
        x: np.array,
        mu_init: Optional[np.ndarray] = None,
    ) -> KMeansResult:
        """Runs the K-Means clustering procedure on the provided dataset.

        Args:
            x (np.array): Dataset array of shape `[n, d]`.
            mu_init (np.ndarray): Centroid initialization array of shape
                `[k, d]`.

        Returns:
            KMeansResult: K-Means clustering result.
        """
        if mu_init is None:
            mu_init = self.init_centroids(x)
        # insert your code here
        mu = mu_init.copy()
        distortion_prev = float("inf")
        membership = np.zeros((x.shape[0], self.config.k), dtype=np.float32)

        for iteration in range(self.config.max_iterations):
            # Assign points to nearest centroid
            distances = np.linalg.norm(
                x[:, np.newaxis] - mu, axis=2
            )  # shape [n, k]
            cluster_assignments = np.argmin(distances, axis=1)

            # Update membership matrix
            membership.fill(0)
            membership[np.arange(x.shape[0]), cluster_assignments] = 1

            # Recompute centroids
            for k_idx in range(self.config.k):
                assigned_points = x[cluster_assignments == k_idx]
                if len(assigned_points) > 0:
                    mu[k_idx] = np.mean(assigned_points, axis=0)

            # Compute distortion
            distortion = compute_distortion(x, mu, membership)

            # Check for convergence
            if abs(distortion_prev - distortion) < self.config.epsilon:
                break
            distortion_prev = distortion

        return KMeansResult(
            config=self.config,
            distortion=distortion,
            mu=mu,
            membership=membership,
        )

    def fit_repeats(
        self,
        x: np.ndarray,
        repeats: int,
    ) -> KMeansResult:
        """Repeats the K-Means cluster procedure by repeatedly calling the fit
        method above. The result with the lowest distortion is then returned.

        Args:
            x (np.array): Dataset array of shape `[n, d]`.
            repeats (int, optional): Number of repeats for which the algorithm is run.

        Returns:
            KMeansResult: Clustering result with the lowest distortion.
        """
        # insert your code here
        best_result = None
        lowest_distortion = float("inf")

        for repeat in range(repeats):
            result = self.fit(x)
            if result.distortion < lowest_distortion:
                lowest_distortion = result.distortion
                best_result = result

        return best_result


@click.command()
@click.option(
    "--filename",
    default="clustering.json",
    type=str,
    help="Cluster result filename.",
)
@click.option("--k", default=10, type=int, help="K parameter.")
@click.option(
    "--max-iterations",
    type=int,
    default=100,
    help="Maximum number of iterations.",
)
@click.option(
    "--epsilon",
    type=float,
    default=1e-3,
    help="Minimum distortion improvement per iteration.",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    help="Number of K-Means clustering repeats.",
)
def main(
    filename: str,
    k: int,
    max_iterations: int,
    epsilon: float,
    repeats: int,
):
    data = MNIST7K.from_file()
    kmeans = KMeans(
        KMeansConfig(
            k=k,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
    )
    kmeans_res = kmeans.fit_repeats(
        data.imgs.reshape(-1, 28 * 28),
        repeats=repeats,
    )
    kmeans_res.to_file(filename)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
