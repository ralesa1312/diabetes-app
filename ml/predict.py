import logging
import os
import sys

import pandas as pd
import yaml

from .utils import MODEL_DIR, load_model

logging.basicConfig(level=logging.INFO)


def run_inference(input_csv_path):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Chargement du meilleur modèle sauvegardé par train.py
    model_path = os.path.join(MODEL_DIR, f"{config['model']['name']}.pkl")

    try:
        pipeline = load_model(model_path)

        logging.info(f"Lecture des nouvelles données : {input_csv_path}")
        new_data = pd.read_csv(input_csv_path)

        # Inférence (Le pipeline Scikit-Learn s'occupe du Scaling automatiquement)
        predictions = pipeline.predict(new_data)
        probabilities = pipeline.predict_proba(new_data)

        # Assemblage des résultats
        results = new_data.copy()
        results["prediction"] = predictions
        # On ajoute la probabilité de la classe prédite
        results["confidence"] = probabilities.max(axis=1)

        output_file = "inference_results.csv"
        results.to_csv(output_file, index=False)
        logging.info(f"Inférence terminée. Résultats sauvegardés dans {output_file}")

    except Exception as e:
        logging.error(f"Erreur lors de l'inférence : {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <chemin_fichier_csv>")
    else:
        run_inference(sys.argv[1])
