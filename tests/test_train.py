import os
import pickle
import pandas as pd
from ml.train import train_model

def test_train_model(tmp_path):
    # Fake dataset
    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "bmi": [22, 27, 31, 29, 33],
        "diabetes": [0, 1, 1, 0, 1]
    })
    data_path = tmp_path / "train_data.csv"
    df.to_csv(data_path, index=False)

    # Output paths
    model_output = tmp_path / "model.pkl"
    scaler_output = tmp_path / "scaler.pkl"

    #  Préprocessed directory
    preprocessed_dir = tmp_path / "preprocessed"

    # Train
    train_model(
        data_path=str(data_path),
        target="diabetes",
        model_output=str(model_output),
        scaler_output=str(scaler_output),
        preprocessed_dir=str(preprocessed_dir)
    )

    # 5️⃣ Assertions
    assert model_output.exists()
    assert scaler_output.exists()
    assert preprocessed_dir.exists()  # dossier créé
