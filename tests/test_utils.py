# tests/test_utils.py
import os
import tempfile

from sklearn.ensemble import RandomForestClassifier

from ml.utils import load_model, save_model


def test_save_load_model():
    # 1. Créer un modèle
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # 2. Dossier temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")

        # 3. Sauvegarde
        save_model(model, path)
        assert os.path.exists(path)

        # 4. Chargement
        loaded_model = load_model(path)

        # 5. Assertions
        assert isinstance(loaded_model, RandomForestClassifier)
        assert loaded_model.get_params() == model.get_params()
