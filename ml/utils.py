import logging
import os
import pickle
from typing import Any

# Racine du projet (on remonte de ml/ vers .)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def save_model(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Objet sauvegardé dans : {path}")


def load_model(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
