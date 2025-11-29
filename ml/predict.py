# ml/predict.py

import pandas as pd

from ml.utils import load_model


def predict_single(model_path: str, scaler_path: str, input_data: dict):
    """
    input_data : dict = {"age": 45, "bmi": 28.4, "glucose": 135, ...}
    """
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    df = pd.DataFrame([input_data])
    X_scaled = scaler.transform(df)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {"prediction": int(prediction), "probability": probability.tolist()}
