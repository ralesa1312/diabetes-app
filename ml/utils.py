# ml/utils.py

import os
from typing import Any

import joblib


def save_model(model: Any, path: str) -> None:
    """Sauvegarde un modèle ML au format .pkl."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f" Modèle sauvegardé dans : {path}")


def load_model(path: str) -> Any:
    """Charge un modèle ML depuis un fichier .pkl."""
    if not os.path.exists(path):
        raise FileNotFoundError(f" Modèle introuvable : {path}")

    print(f"Chargement du modèle : {path}")
    return joblib.load(path)
