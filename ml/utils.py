# ml/utils.py (Corrigé pour utiliser logging)

import os
import pickle
import logging
from typing import Any, Dict


# -----------------------------------
# CONFIGURATION GLOBALE DES CHEMINS
# -----------------------------------

# Répertoire racine du projet (ex: 'project_root/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# Répertoire où les modèles sont sauvegardés (ex: 'project_root/models/')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models') 
# Nom du fichier contenant le modèle et le scaler
MODEL_FILE_NAME = "modele_diabete_final_rf.pkl"
# Chemin complet par défaut pour la sauvegarde/chargement
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

def save_model(model: Any, path: str) -> None:
    """Sauvegarde un modèle ML ou un dictionnaire au format .pkl en utilisant pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)
    

    logging.info(f"Modèle sauvegardé dans : {path}")


def load_model(path: str) -> Dict[str, Any]:
    """Charge le dictionnaire contenant le modèle et le scaler depuis un fichier .pkl."""
    if not os.path.exists(path):
        raise FileNotFoundError(f" Modèle introuvable : {path}")


    logging.info(f"Chargement des données du modèle : {path}")
    
    with open(path, "rb") as f:
        # pickle.load charge le dictionnaire {"model": ..., "scaler": ...}
        return pickle.load(f)