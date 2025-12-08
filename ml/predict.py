# ml/predict.py (Version Corrigée, en accord avec preprocess.py)

import pandas as pd
from ml.preprocess import preprocess_data
from ml.utils import load_model 


def predict_single(model_path: str, scaler_path: str, input_data: dict):
    """
    input_data : dict = {"age": 45, "bmi": 28.4, "glucose": 135, ...}
    """
    model = load_model(model_path)
    scaler = load_model(scaler_path)

    # 1. Préparer les données pour le prétraitement/scaling
    df = pd.DataFrame([input_data])

    # 2. **Correction majeure de la cohérence :** # Recréer le jeu de features X de la même manière que dans preprocess_data,
    # SANS inclure la cible (puisqu'on prédit la cible) et après nettoyage initial.
    
    # Simuler le nettoyage et la sélection des features X
    # Si preprocess_data ne fait que dropna() et drop(target), nous devons simuler cela.
    df = df.dropna() # Simuler le dropna() de preprocess_data (très peu d'effet ici)
    
    # Nous assumons que 'target' n'est PAS dans input_data
    # Si le target n'est pas dans input_data, X est simplement le DataFrame:
    X_unscaled = df 
    
    # Si par erreur target est dans input_data (rare pour une prédiction), décommenter :
    # target_placeholder = "target_col_name" # Utilisez le vrai nom de la cible
    # X_unscaled = df.drop(columns=[target_placeholder], errors='ignore')
    
    # 3. Mise à l'échelle : Appliquer le scaler sur X, tel que preprocess_data l'a produit
    X_scaled = scaler.transform(X_unscaled)

    # 4. Prédiction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {"prediction": int(prediction), "probability": probability.tolist()}