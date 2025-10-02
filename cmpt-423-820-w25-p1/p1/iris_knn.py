from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def count_species_index(
    species_idxs: np.ndarray,
    num_species: int = 3,
) -> np.ndarray:
    """This method counts how often a species index occurs in each row of the
    provided species index prediction matrix. The input matrix `species_idx`
    is of shape `[n, k]`. Every entry in species_idxs must be at least 0 and
    at most `num_species - 1`.

    For example, if `species_idxs` is set to

    array([[0, 1, 1],
           [1, 2, 1],
           [0, 0, 0]])

    then this function returns an array

    array([[1, 2, 0],
           [0, 2, 1],
           [3, 0, 0]])

    Here, entry `[i, j]` of the returned tensor is set to how often the class
    `j` is contained in row `i`.

    Args:
        species_idxs (np.ndarray): Species index matrix of shape `[n, k]`.
        num_species (int, optional): Number of species. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    assert species_idxs.min() >= 0 and species_idxs.max() < num_species
    pass  # Insert your code here.


def distance_matrix_from_features(
    feat1: np.ndarray, feat2: np.ndarray
) -> np.ndarray:
    """Computes the distance matrix for every pair of row vectors in `feat1`
    and `feat2`. The matrix `feat1` has shape `[n, d]` and the matrix `feat2`
    has shape `[m, d]`. The resulting distance matrix then has shape `[n, m]`.
    The entry `(i, j)` of the returned distance matrix is equal to the L2 distance
    (euclidean disance) between `feat1[i]` and `feat2[j]`.

    Args:
        feat1 (np.ndarray): Input feature tensor of shape `[n, d]`
        feat2 (np.ndarray): Input feature tensor of shape `[m, d]`

    Returns:
        np.ndarray: Distance matrix of shape `[n, m]`.
    """
    assert len(feat1.shape) == 2
    assert len(feat2.shape) == 2
    assert feat1.shape[1] == feat2.shape[1]
    pass  # Insert your code here.


def distance_matrix_to_knn_indices(
    distance_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """Find the k nearest neighbours from the provided distance matrix.
    This method shorts every row of this matrix to finds the indices of
    the k nearest training points.

    Args:
        distance_matrix (np.ndarray): Distance matrix as outputted by
            `get_distance_matrix`.
        k (int): Number of neighbours to return.

    Returns:
        np.ndarray: k nearest neighbours matrix of shape `[n, k]`
    """
    pass  # Insert your code here.


@dataclass
class IrisData:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def all(self) -> pd.DataFrame:
        return pd.concat((self.train, self.test))

    @property
    def species_names(self) -> Tuple[str, ...]:
        return ("versicolor", "virginica", "setosa")

    def species_to_index(self, species: str) -> int:
        return {n: i for i, n in enumerate(self.species_names)}[species]

    @staticmethod
    def from_json(filename: str) -> "IrisData":
        iris_data = pd.read_json(filename)
        iris_train = iris_data[iris_data["set_name"] == "train"]
        iris_test = iris_data[iris_data["set_name"] == "test"]
        return IrisData(
            train=iris_train.drop(columns=["set_name"]),
            test=iris_test.drop(columns=["set_name"]),
        )

    @property
    def feature_columns(self) -> List[str]:
        return ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    @property
    def train_features(self) -> np.ndarray:
        return self.train[self.feature_columns].values

    @property
    def test_features(self) -> np.ndarray:
        return self.test[self.feature_columns].values

    @property
    def train_species_idxs(self) -> np.ndarray:
        species_str = self.train["species"].values
        species_idx = [self.species_to_index(s) for s in species_str]
        return np.array(species_idx, dtype=np.long)

    @property
    def test_species_idxs(self) -> np.ndarray:
        species_str = self.test["species"].values
        species_idx = [self.species_to_index(s) for s in species_str]
        return np.array(species_idx, dtype=np.long)

    def knn_indices_to_species_indices(
        self, train_idx: np.ndarray
    ) -> np.ndarray:
        """Maps indices of k-nearest neighbours to species indices. The
        input tensor `train_idx` can have any shape and the returned species
        tensor should have the same shape as `train_idx`. Every entry of
        `train_idx` indexes into a row of `self.train_features`.

        Args:
            train_idx (np.ndarray): k-nearest neighbor index matrix.

        Returns:
            np.ndarray: Species index matrix.
        """
        species_idxs = self.train_species_idxs[train_idx.reshape(-1)]
        return species_idxs.reshape(*train_idx.shape)

    def predict_species_knn(self, features: np.ndarray, k: int) -> np.ndarray:
        """Implements k-NN lookup for the provided features.

        Args:
            features (np.ndarray): Feature matrix of shape `[n, 4]`.
            k (int): k value

        Returns:
            np.ndarray: Prediced species index or shape `[n,]`.
        """
        pass  # Insert your code here.


def confusion_matrix(
    idx_pred: np.ndarray,
    idx_true: np.ndarray,
) -> np.ndarray:
    """Computes a confusion matrix.

    Args:
        idx_pred (np.ndarray): Predicted class index, shape `[n,]`, dtype np.long.
        idx_true (np.ndarray): True class index, shape `[n,]`, dtype np.long.

    Returns:
        np.ndarray: Confusion matrix of shape `[n,n]`.
    """
    # assert idx_pred.dtype == np.long
    # assert idx_true.dtype == np.long
    # The following code scales class indices to range from 0 to num_labels - 1.
    idx_all = np.concat((idx_pred, idx_true))
    idx_pred = idx_pred - idx_all.min()
    idx_true = idx_true - idx_all.min()
    num_labels = idx_all.max() - idx_all.min() + 1
    # Implementation of confusion matrix.
    pass  # Insert your code here.
