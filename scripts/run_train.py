import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)  # ajoute le projet au PYTHONPATH

from ml.train import train_model
from ml.utils import DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH

PREPROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

if __name__ == "__main__":
    train_model(
        data_path=os.path.join(DATA_DIR, "raw/diabetes_012_health_indicators_BRFSS2015.csv"),
        target="Diabetes_012",
        model_output=DEFAULT_MODEL_PATH,
        scaler_output=DEFAULT_SCALER_PATH,
        preprocessed_dir=PREPROCESSED_DIR
    )
