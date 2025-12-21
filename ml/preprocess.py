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
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler


def split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    X_train_path: str = None,
    X_test_path: str = None,
    y_train_path: str = None,
    y_test_path: str = None,
):
    """
    Sépare les données en train/test et sauvegarde les fichiers si les chemins sont fournis.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Sauvegarde si chemins fournis
    if X_train_path:
        pd.DataFrame(X_train).to_csv(X_train_path, index=False)
    if X_test_path:
        pd.DataFrame(X_test).to_csv(X_test_path, index=False)
    if y_train_path:
        pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    if y_test_path:
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)

    return X_train, X_test, y_train, y_test
