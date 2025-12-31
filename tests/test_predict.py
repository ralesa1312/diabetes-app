import os

import pandas as pd

from ml.predict import run_inference


def test_predict_output_format(tmp_path):
    # Créer un petit fichier CSV factice pour le test
    dummy_data = pd.DataFrame(
        {
            "HighBP": [1.0, 0.0],
            "HighChol": [1.0, 0.0],
            "CholCheck": [1.0, 1.0],
            "BMI": [30.0, 22.0],
            "Smoker": [1.0, 0.0],
            "Stroke": [0.0, 0.0],
            "HeartDiseaseorAttack": [0.0, 0.0],
            "PhysActivity": [0.0, 1.0],
            "Fruits": [0.0, 1.0],
            "Veggies": [1.0, 1.0],
            "HvyAlcoholConsump": [0.0, 0.0],
            "AnyHealthcare": [1.0, 1.0],
            "NoDocbcCost": [0.0, 0.0],
            "GenHlth": [3.0, 1.0],
            "MentHlth": [0.0, 0.0],
            "PhysHlth": [0.0, 0.0],
            "DiffWalk": [0.0, 0.0],
            "Sex": [1.0, 0.0],
            "Age": [9.0, 4.0],
            "Education": [4.0, 6.0],
            "Income": [3.0, 8.0],
        }
    )

    input_path = tmp_path / "test_input.csv"
    dummy_data.to_csv(input_path, index=False)

    run_inference(str(input_path))

    assert os.path.exists("inference_results.csv")
    results = pd.read_csv("inference_results.csv")
    assert "prediction" in results.columns
    assert "confidence" in results.columns
