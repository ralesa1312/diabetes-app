import logging

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

from ml.preprocess import preprocess_data, split_data
from ml.utils import save_model


def train_model(
    data_path: str,
    target: str,
    model_output: str,
    scaler_output: str,
    preprocessed_dir: str,  # dossier pour X_train/X_test
) -> None:
    """
    Entraîne un modèle de classification RandomForest et sauvegarde
    le modèle, le scaler et les datasets train/test.
    """

    # 1) Charger données
    df = pd.read_csv(data_path)
    logging.info(
        f"Données chargées depuis {data_path} avec {df.shape[0]} lignes et {df.shape[1]} colonnes."
    )

    # 2) Prétraiter données
    X, y, scaler = preprocess_data(df, target)
    logging.info("Données prétraitées.")

    if preprocessed_dir is None:
        from ml.utils import DATA_DIR
        preprocessed_dir = os.path.join(DATA_DIR, "processed")

    os.makedirs(preprocessed_dir, exist_ok=True)  # <-- ici, on crée le dossier

    # 3) Séparer train/test et sauvegarder les fichiers
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=0.2,
        random_state=42,
        X_train_path=os.path.join(preprocessed_dir, "X_train.csv"),
        X_test_path=os.path.join(preprocessed_dir, "X_test.csv"),
        y_train_path=os.path.join(preprocessed_dir, "y_train.csv"),
        y_test_path=os.path.join(preprocessed_dir, "y_test.csv"),
    )

    logging.info(
        f"Données séparées et sauvegardées dans {preprocessed_dir} "
        f"(train {X_train.shape[0]}, test {X_test.shape[0]})."
    )

    # 4) Entraîner modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    logging.info("Modèle entraîné.")

    # 5) Sauvegarde modèle et scaler
    save_model(model, model_output)
    save_model(scaler, scaler_output)

    logging.info(f"Modèle sauvegardé dans {model_output}")
    logging.info(f"Scaler sauvegardé dans {scaler_output}")
