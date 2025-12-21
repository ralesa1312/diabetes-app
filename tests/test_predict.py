# ml/predict.py
import pandas as pd

from ml.utils import load_model


def predict(model_path: str, scaler_path: str, data_path: str):
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    df = pd.read_csv(data_path)
    X_scaled = scaler.transform(df)

    return model.predict(X_scaled)
