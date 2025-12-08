# tests/test_predict.py
import os
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.predict import predict_single
from ml.utils import save_model


def test_predict_single():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    model.fit(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(len(X_scaled))

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        scaler_path = os.path.join(tmpdir, "scaler.pkl")
        save_model(model, model_path)
        save_model(scaler, scaler_path)

        input_data = {"feature1": 2, "feature2": 3}
        output = predict_single(model_path, scaler_path, input_data)
        assert "prediction" in output
        assert "probability" in output
        assert len(output["probability"]) == 2
