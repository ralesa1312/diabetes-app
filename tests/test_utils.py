import os
import pytest
from ml.utils import save_model, load_model, DATA_DIR, MODEL_DIR

def test_save_and_load_model(tmp_path):
    """Vérifie qu'un objet sauvegardé est identique une fois rechargé."""
    test_obj = {"key": "value", "list": [1, 2, 3]}
    test_path = os.path.join(tmp_path, "test_object.pkl")
    
    # Test sauvegarde
    save_model(test_obj, test_path)
    assert os.path.exists(test_path)
    
    # Test chargement
    loaded_obj = load_model(test_path)
    assert loaded_obj == test_obj
    assert loaded_obj["key"] == "value"

def test_load_model_file_not_found():
    """Vérifie que load_model lève bien une erreur si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_model("path/to/non_existent_file.pkl")

def test_paths_definitions():
    """Vérifie que les chemins de base sont correctement définis."""
    assert "data" in DATA_DIR
    assert "models" in MODEL_DIR