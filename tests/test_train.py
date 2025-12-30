import os
from ml.train import train
from ml.utils import MODEL_DIR
import yaml

def test_train_produces_model():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name'] + ".pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Supprimer l'ancien modèle s'il existe pour être sûr du test
    if os.path.exists(model_path):
        os.remove(model_path)
        
    train()
    
    assert os.path.exists(model_path)