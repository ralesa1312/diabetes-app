import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.evaluate import evaluate


def test_evaluate(tmp_path):
    # ---------------------------
    # 1. Fake dataset
    # ---------------------------
    df = pd.DataFrame({"age": [25, 35, 45, 55], "bmi": [23, 28, 31, 29], "diabetes": [0, 1, 1, 0]})

    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)

    # ---------------------------
    # 2. Fake scaler
    # ---------------------------
    scaler = StandardScaler()
    X = df.drop(columns=["diabetes"])
    scaler.fit(X)

    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ---------------------------
    # 3. Fake model
    # ---------------------------
    model = RandomForestClassifier(random_state=42)
    model.fit(scaler.transform(X), df["diabetes"])

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ---------------------------
    # 4. Run evaluation
    # ---------------------------
    metrics_dir = tmp_path / "metrics"

    metrics = evaluate(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        data_path=str(data_path),
        target="diabetes",
        metrics_dir=str(metrics_dir),
    )

    # ---------------------------
    # 5. Assertions
    # ---------------------------
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics

    assert os.path.exists(metrics_dir / "metrics.csv")
    assert os.path.exists(metrics_dir / "confusion_matrix.csv")
