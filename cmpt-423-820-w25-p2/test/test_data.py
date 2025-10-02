def test_formula_data():
    import numpy as np

    from p2.formula import FormulaData

    data = FormulaData()
    assert data.train_imgs.shape == (200000, 28, 28 * 3)
    assert data.test_imgs.shape == (10000, 28, 28 * 3)
    assert data.train_labels_str.shape == (200000, 3)
    assert data.test_labels_str.shape == (10000, 3)
    assert data.train_imgs.dtype == np.float32
    assert data.test_imgs.dtype == np.float32
