def test_kmeans_result_persistance():
    """Tests persistance of the KMeans result class."""
    import os

    import numpy as np

    from kmeans.kmeans import KMeansConfig, KMeansResult

    res = KMeansResult(
        config=KMeansConfig(k=10),
        distortion=1.0,
        mu=np.random.uniform((5, 10)),
        membership=np.random.uniform((50, 10)),
    )
    res.to_file("test_res.json")

    res_recon = KMeansResult.from_file("test_res.json")
    assert res.config.k == res_recon.config.k
    assert res.config.max_iterations == res_recon.config.max_iterations
    assert res.config.epsilon == res_recon.config.epsilon
    assert res.distortion == res_recon.distortion
    assert np.allclose(res.mu, res_recon.mu)
    assert np.allclose(res.membership, res_recon.membership)
    os.remove("test_res.json")
