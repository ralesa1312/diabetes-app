import os
import pandas as pd
import pytest
from ml.preprocess import preprocess
from ml.utils import DATA_DIR

def test_preprocess_creates_files():
    # Exécuter le preprocess
    preprocess()
    
    train_path = os.path.join(DATA_DIR, "processed", "train.parquet")
    test_path = os.path.join(DATA_DIR, "processed", "test.parquet")
    
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

def test_class_balancing():
    train_path = os.path.join(DATA_DIR, "preprocessed", "train.parquet")
    df = pd.read_parquet(train_path)
    
    counts = df["Diabetes_012"].value_counts()
    # Vérifier que les classes sont égales à la target_size (35346)
    assert counts.min() == 35346
    assert counts.max() == 35346