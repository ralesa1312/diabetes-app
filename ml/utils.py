# ml/utils.py
import logging
import os
import pickle
from typing import Any

# -----------------------------------
# CONFIGURATION GLOBALE DES CHEMINS
# -----------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Fichiers par défaut (optionnels)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DEFAULT_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# -----------------------------------
# FONCTIONS I/O MODELS
# -----------------------------------


def save_model(obj: Any, path: str) -> None:
    """Sauvegarde un objet Python (modèle, scaler, dict, etc.) en .pkl."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    logging.info(f"Objet sauvegardé dans : {path}")


def load_model(path: str) -> Any:
    """Charge un objet Python depuis un fichier .pkl."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    logging.info(f"Chargement de l'objet depuis : {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
