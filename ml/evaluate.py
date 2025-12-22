# ml/evaluate.py
import logging
import os
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.utils import load_model


def evaluate(
    model_path: str,
    scaler_path: str,
    data_path: str,
    metrics_dir: str,
):
    """
    Évalue un modèle sur les données de test préprocessées
    et sauvegarde les métriques dans data/metrics/
    """

    os.makedirs(metrics_dir, exist_ok=True)

    # -------------------------
    # 1. Charger modèle & scaler
    # -------------------------
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    # -------------------------
    # 2. Charger données TEST
    # -------------------------
    X_test = pd.read_csv(os.path.join(data_path, "processed/X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "processed/y_test.csv")).squeeze()

    # -------------------------
    # 3. Scaling (transform ONLY)
    # -------------------------
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # 4. Prédictions
    # -------------------------
    preds = model.predict(X_test_scaled)

    # -------------------------
    # 5. Métriques
    # -------------------------
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    logging.info("Accuracy : %.4f", accuracy)
    logging.info("F1-score : %.4f", f1)
    logging.info("Matrice de confusion :\n%s", cm)

    # -------------------------
    # 6. Sauvegarde métriques
    # -------------------------
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "f1_macro": f1
    }])
    metrics_df.to_csv(
        os.path.join(metrics_dir, "metrics.csv"),
        index=False
    )

    pd.DataFrame(cm).to_csv(
        os.path.join(metrics_dir, "confusion_matrix.csv"),
        index=False
    )

    logging.info("Métriques sauvegardées dans %s", metrics_dir)

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "confusion_matrix": cm
    }
