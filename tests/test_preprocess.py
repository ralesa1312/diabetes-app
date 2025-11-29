# tests/test_preprocess.py
import pandas as pd
import numpy as np
from ml.preprocess import preprocess_data, split_data

def test_preprocess_data():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0]
    })
    X, y, scaler = preprocess_data(df, target="target")
    
    assert X.shape == (3, 2)
    assert y.shape[0] == 3

def test_split_data():
    X = np.array([[1],[2],[3],[4]])
    y = np.array([0,1,0,1])
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)
    
    assert len(X_train) == 2
    assert len(X_test) == 2
