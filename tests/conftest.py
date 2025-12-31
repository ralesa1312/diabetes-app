import os
import sys

import pandas as pd
import pytest

# Ajoute la racine du projet au chemin de recherche de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Crée les dossiers nécessaires avant les tests et nettoie après."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/preprocessed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Création d'un mini dataset de test si le fichier raw n'existe pas
    raw_path = "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"
    if not os.path.exists(raw_path):
        dummy_df = pd.DataFrame(
            {
                "Diabetes_012": [0.0, 1.0, 2.0] * 100,
                "HighBP": [0.0] * 300,
                "BMI": [25.0] * 300,
                # ... ajoutez d'autres colonnes au besoin pour matcher votre config
            }
        )
        dummy_df.to_csv(raw_path, index=False)

    yield  # Ici, les tests s'exécutent

    # shutil.rmtree("data/preprocessed")
