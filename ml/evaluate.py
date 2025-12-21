# ml/evaluate.py
import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.preprocess import preprocess_data
from ml.utils import load_model


def evaluate(
    model_path: str,
    scaler_path: str,
    data_path: str,
    target: str,
    metrics_dir: str,
):
    """
    Évalue un modèle sur des données et sauvegarde les métriques dans data/metrics/
    """

    os.makedirs(metrics_dir, exist_ok=True)

    #  Charger modèle et scaler
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    #  Charger données
    df = pd.read_csv(data_path)

    #  Prétraiter données
    X, y, _ = preprocess_data(df, target)
    X_scaled = scaler.transform(X)

    #  Prédictions
    preds = model.predict(X_scaled)

    # Calcul des métriques
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    cm = confusion_matrix(y, preds)

    logging.info("Accuracy : %.4f", accuracy)
    logging.info("F1-score : %.4f", f1)
    logging.info("Matrice de confusion :\n%s", cm)

    # Sauvegarde métriques (DATA, pas MODEL)
    metrics_df = pd.DataFrame([{"accuracy": accuracy, "f1_macro": f1}])
    metrics_df.to_csv(os.path.join(metrics_dir, "metrics.csv"), index=False)

    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"), index=False)

    logging.info(f"Métriques sauvegardées dans {metrics_dir}")

    return {"accuracy": accuracy, "f1_macro": f1, "confusion_matrix": cm}
