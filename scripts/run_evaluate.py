import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)  # ajoute le projet au PYTHONPATH

from ml.evaluate import evaluate
from ml.utils import DATA_DIR, MODEL_DIR, DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH

# Dossier pour sauvegarder les métriques
METRICS_DIR = os.path.join(DATA_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(METRICS_DIR, "metrics.csv")

if __name__ == "__main__":
    evaluate(
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=DEFAULT_SCALER_PATH,
        data_path=os.path.join(DATA_DIR, "processed/X_test.csv"),
        target="Diabetes_012",
        metrics_dir=METRICS_FILE
    )
