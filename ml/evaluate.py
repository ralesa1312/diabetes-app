# ml/evaluate.py
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.preprocess import preprocess_data
from ml.utils import load_model


def evaluate(model_path: str, scaler_path: str, data_path: str, target: str):
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    df = pd.read_csv(data_path)
    X, y, _ = preprocess_data(df, target)

    X_scaled = scaler.transform(df.drop(columns=[target]))

    preds = model.predict(X_scaled)

    logging.info("Accuracy : %f", accuracy_score(y, preds))
    logging.info("F1-score : %f", f1_score(y, preds, average="macro"))
    logging.info("Matrice de confusion :\n%s", confusion_matrix(y, preds))
