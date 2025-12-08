import logging

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

from ml.preprocess import preprocess_data, split_data


def train_model(data_path: str, target: str, model_output: str) -> None:
    """
    Entraîne un modèle de classification RandomForest et le sauvegarde.
    """
    # 1) Charger données
    df = pd.read_csv(data_path)
    logging.info(
        f"Données chargées depuis {data_path} avec {df.shape[0]} lignes et {df.shape[1]} \
                 colonnes."
    )

    # 2) Prétraiter données
    X, y, scaler = preprocess_data(df, target)
    logging.info("Données prétraitées.")

    # 3) Séparer train/test
    X_train, X_test, y_train, y_test = split_data(X, y)
    logging.info(
        f"Données séparées en train ({X_train.shape[0]} échantillons) et \
                 test ({X_test.shape[0]} échantillons)."
    )

    # 4) Entraîner modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    logging.info("Modèle entraîné.")

    with open(model_output, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    logging.info(f"Modèle et scaler sauvegardés dans {model_output}.")

