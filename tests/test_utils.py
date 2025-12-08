# tests/test_utils.py
import os
import tempfile

from sklearn.ensemble import RandomForestClassifier

from ml.utils import load_model, save_model


def test_save_load_model():
    model = RandomForestClassifier(n_estimators=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        save_model(model, path)
        assert os.path.exists(path)

        loaded_model = load_model(path)
        assert isinstance(loaded_model) == RandomForestClassifier
        assert loaded_model.get_params() == model.get_params()
