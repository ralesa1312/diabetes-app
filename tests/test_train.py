# tests/test_train.py
import os
import pandas as pd
import tempfile
from ml.train import train_model

def test_train_model():
    df = pd.DataFrame({
        "age": [25, 30, 45],
        "glucose": [120, 140, 130],
        "bmi": [22, 27, 28],
        "target": [0, 1, 0]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        model_path = os.path.join(tmpdir, "rf.pkl")
        df.to_csv(csv_path, index=False)

        model, scaler = train_model(csv_path, target="target", model_output=model_path)
        assert os.path.exists(model_path)
