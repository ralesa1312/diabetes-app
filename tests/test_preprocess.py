import numpy as np
import pandas as pd

from ml.preprocess import preprocess_data, split_data


def test_preprocess_data():
    df = pd.DataFrame({"age": [20, 30, 40], "bmi": [22.5, 27.1, 30.2], "diabetes": [0, 1, 1]})

    X_scaled, y, scaler = preprocess_data(df, target="diabetes")

    # X : 2 features
    assert X_scaled.shape == (3, 2)

    # y
    assert list(y) == [0, 1, 1]

    # scaler
    assert scaler is not None

    # scaling centré
    np.testing.assert_almost_equal(X_scaled.mean(axis=0), [0, 0], decimal=6)


def test_split_data():
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2
