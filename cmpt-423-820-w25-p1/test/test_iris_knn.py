def test_distance_matrix_from_features():
    """Tests `distance_matrix_from_features`.

    This unit test uses two small example feature matrices to test the distance
    matrix calculation function. These two example matrices have different
    sizes than the matrices used for the Iris dataset, but they are picked to
    be small and easy to understand to quickly spot any bugs in the
    `distance_matrix_from_features` function.

    The `train_feat` contains four row feature vectors and each feature vector
    has dimension two. So the shape of the `train_feat` matrix is size
    `[4, 2]`. The `test_feat` contains two row feature vectors and each
    feature vector has dimension two. So the shape of the `train_feat` matrix
    is size `[2, 2]`. The shape of the distance matrix is then `[2, 4]`, where
    the entry `[i, j]` (ith row and jth column) is set to the euclidean
    distance between the ith test feature vector and the jth training feature
    vector.

    The following unit test tests if the `distance_matrix_from_features`
    returns the correct distance matrix.
    """
    import numpy as np

    from p1.iris_knn import distance_matrix_from_features

    train_feat = np.array(
        [
            [0.0, 0.0],  # Point A
            [1.0, 0.0],  # Point B
            [0.0, 1.0],  # Point C
            [1.0, 1.0],  # Point D
        ],
        dtype=np.float32,
    )
    test_feat = np.array(
        [
            [0.1, 0.1],  # Point E
            [0.0, 0.9],  # Point F
        ],
        dtype=np.float32,
    )

    d_00 = np.sqrt(0.1**2 + 0.1**2)  # Distance btw. A and E
    d_01 = np.sqrt(0.9**2 + 0.1**2)  # Distance btw. B and E
    d_02 = np.sqrt(0.1**2 + 0.9**2)  # Distance btw. C and E
    d_03 = np.sqrt(0.9**2 + 0.9**2)  # Distance btw. D and E
    d_10 = np.sqrt(0.0**2 + 0.9**2)  # Distance btw. A and F
    d_11 = np.sqrt(1.0**2 + 0.9**2)  # Distance btw. B and F
    d_12 = np.sqrt(0.0**2 + 0.1**2)  # Distance btw. C and F
    d_13 = np.sqrt(1.0**2 + 0.1**2)  # Distance btw. D and F
    dist_mat_corr = np.array(
        [
            [d_00, d_01, d_02, d_03],
            [d_10, d_11, d_12, d_13],
        ],
        dtype=np.float32,
    )

    dist_mat = distance_matrix_from_features(test_feat, train_feat)
    assert dist_mat.shape == dist_mat_corr.shape
    # np.allclose is used to account for floating point round-off errors.
    assert np.allclose(dist_mat, dist_mat_corr)


def test_distance_matrix_to_knn_indices():
    """Tests `distance_matrix_to_knn_indices`.

    Given a distance matrix, the `distance_matrix_to_knn_indices` function
    finds which k training feature vectors are closes to each test feature
    vector. Each entry `[i, j]` in the distance matrix corresponds to the
    distance between the ith test feature vector and the jth training feature
    vector. To find the k training feature vectors that are closes to the ith
    test feature vectors, the function `distance_matrix_to_knn_indices`
    computes the column indices of the k lowest entries for each row. The
    resulting index matrix has then shape `[n, k]` where `n` is the number of
    test feature vectors.

    The test code below uses the same `[2, 4]` distance matrix from the example
    in the unit test `test_distance_matrix_from_features`. The index matrix
    returned by `distance_matrix_to_knn_indices` should then have shape
    `[2, 3]` for k=3.
    """
    import numpy as np

    from p1.iris_knn import distance_matrix_to_knn_indices

    # This test uses the same example from the unit test
    # test_distance_matrix_from_features
    # Point A = [0.0, 0.0], in col idx 0
    # Point B = [1.0, 0.0], in col idx 1
    # Point C = [0.0, 1.0], in col idx 2
    # Point D = [1.0, 1.0], in col idx 3
    # Point E = [0.1, 0.1], in row idx 0
    # Point F = [0.0, 0.9], in row idx 1
    d_00 = np.sqrt(0.1**2 + 0.1**2)  # Distance btw. A and E
    d_01 = np.sqrt(0.9**2 + 0.1**2)  # Distance btw. B and E
    d_02 = np.sqrt(0.1**2 + 0.9**2)  # Distance btw. C and E
    d_03 = np.sqrt(0.9**2 + 0.9**2)  # Distance btw. D and E
    d_10 = np.sqrt(0.0**2 + 0.9**2)  # Distance btw. A and F
    d_11 = np.sqrt(1.0**2 + 0.9**2)  # Distance btw. B and F
    d_12 = np.sqrt(0.0**2 + 0.1**2)  # Distance btw. C and F
    d_13 = np.sqrt(1.0**2 + 0.1**2)  # Distance btw. D and F
    dist_mat = np.array(
        [
            [d_00, d_01, d_02, d_03],
            [d_10, d_11, d_12, d_13],
        ],
        dtype=np.float32,
    )
    knn_idx_mat = distance_matrix_to_knn_indices(dist_mat, k=3)
    knn_idx_mat_corr = np.array(
        [
            [0, 1, 2],  # Closest points to E: A < B < C
            [2, 0, 3],  # Closest points to F: C < A < D
        ],
        dtype=np.long,
    )
    assert (knn_idx_mat == knn_idx_mat_corr).all()


def test_count_species_index():
    """Tests `count_species_index`

    The function `count_species_index` computes which species index is most
    common amongst the k nearest neighbors. As input, this function receives an
    index matrix of shape `[n, k]`, where `n` is the number of test feature
    vectors and k is the neighourhood size. The entry `[i, j]` in this matrix
    equals the species index of the jth neighbour of the ith test feature
    vector.

    The test code below tests `count_species_index` the function for a small
    `[2, 4]` example (two test feature vectors with a neighbourhood size of
    k=4). Becaise there are three different species, the returned matrix has
    shape `[2, 3]`, there each entry `[i, j]` of the returned matrix
    corresponds to how often a species j occurs in the neighbourhood of the ith
    test feature vector.
    """
    import numpy as np

    from p1.iris_knn import count_species_index

    species_neighborhood = np.array(
        [
            [0, 1, 1, 1],
            [1, 2, 1, 0],
        ],
        dtype=np.long,
    )
    species_counts = count_species_index(species_neighborhood, num_species=3)
    species_counts_corr = np.array(
        [
            [1, 3, 0],
            [1, 2, 1],
        ],
        dtype=np.long,
    )
    assert (species_counts == species_counts_corr).all()


def test_confusion_matrix():
    """Tests `confusion_matrix`

    The following code tests the `confusion_matrix` calculation with two toy
    examples. The first example tests a perfect classification of three
    different classes. Therefore, the returned `confusion_matrix` has shape
    `[3, 3]`. Because each class occurrs only once in the predicted and true
    class label arrays (`idx_pred` and `idx_true`), the confusion matrix is
    equal to the identity matrix.

    The second example tests if two predictions are incorrect and how these
    incorrect predictions change the confusion matrix. In this example, instead
    of predicting class index `2` the class index `1` is predicted three times
    and therefore entry `[2, 1]` is set to three in the resulting confusion
    matrix.
    """
    import numpy as np

    from p1.iris_knn import confusion_matrix

    # This first test uses three correct predictions.
    idx_pred = np.array([0, 1, 2], dtype=np.long)
    idx_true = np.array([0, 1, 2], dtype=np.long)
    conf_mat_cnt = confusion_matrix(idx_pred, idx_true)
    conf_mat_cnt_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    assert (conf_mat_cnt == conf_mat_cnt_true).all()

    # This second test uses six predictions, three correct and three incorrect.
    idx_pred = np.array([0, 1, 2, 1, 1, 1], dtype=np.long)
    idx_true = np.array([0, 1, 2, 2, 2, 2], dtype=np.long)
    conf_mat_cnt = confusion_matrix(idx_pred, idx_true)
    conf_mat_cnt_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 3, 1],
        ],
        dtype=np.int32,
    )
    assert (conf_mat_cnt == conf_mat_cnt_true).all()
