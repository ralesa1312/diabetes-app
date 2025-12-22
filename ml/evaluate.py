# ml/evaluate.py
import logging
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

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
    X_test = pd.read_csv(os.path.join(data_path, "processed/X_test.csv"), sep=",")
    y_test = pd.read_csv(os.path.join(data_path, "processed/y_test.csv"), sep=",").squeeze()

    # -------------------------
    # 3. Scaling (transform ONLY)
    # -------------------------
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # 4. Prédictions
    # -------------------------
    y_pred = model.predict(X_test_scaled)

    # Probabilités pour AUC
    y_proba = model.predict_proba(X_test_scaled)

    # -------------------------
    # 5. Métriques
    # -------------------------
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # Multi-class AUC (ONE VS REST)
    auc_score = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    cm = confusion_matrix(y_test, y_pred)

    logging.info("Accuracy : %.4f", accuracy)
    logging.info("F1-score : %.4f", f1)
    logging.info("AUC : %.4f", auc_score)
    logging.info("Matrice de confusion :\n%s", cm)

    # -------------------------
    # 6. Sauvegarde métriques
    # -------------------------
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "f1_macro": f1,
        "auc": auc_score
    }])

    metrics_df.to_csv(
        os.path.join(metrics_dir, "metrics.csv"),
        index=False
    )

    pd.DataFrame(cm).to_csv(
        os.path.join(metrics_dir, "confusion_matrix.csv"),
        index=False
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "auc": auc_score,
        "confusion_matrix": cm
    }
