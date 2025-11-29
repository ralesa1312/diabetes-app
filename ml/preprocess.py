# ml/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame, target: str):
    """
    Nettoyage + séparation X / y + scaling.
    """
    df = df.copy()
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Sépare les données en train/test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
