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
    
    # 1. Prétraiter les données brutes. 
    # X contient les features encodées, prêtes pour la mise à l'échelle.
    # y contient les étiquettes cibles.
    X, y, _ = preprocess_data(df, target) 

    #  Appliquer le scaler sur X, et non sur le DataFrame brut.
    # Cette étape garantit que le scaler voit les colonnes dans le même format et ordre 
    # que lors de l'entraînement.
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    logging.info("Accuracy : %f", accuracy_score(y, preds))
    logging.info("F1-score : %f", f1_score(y, preds, average="macro"))
    logging.info("Matrice de confusion :\n%s", confusion_matrix(y, preds))