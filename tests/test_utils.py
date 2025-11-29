# tests/test_utils.py
import os
import tempfile
import pytest
from ml.utils import save_model, load_model
from sklearn.ensemble import RandomForestClassifier

def test_save_load_model():
    model = RandomForestClassifier(n_estimators=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        save_model(model, path)
        assert os.path.exists(path)
        
        loaded_model = load_model(path)
        assert type(loaded_model) == RandomForestClassifier
        assert loaded_model.get_params() == model.get_params()