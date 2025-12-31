import logging
import os
from collections import Counter

import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from .utils import DATA_DIR

logging.basicConfig(level=logging.INFO)


def preprocess():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_path = os.path.join(DATA_DIR, "raw", config["data"]["raw_file"])
    df = pd.read_csv(raw_path).drop_duplicates()

    # 1. Séparation Train/Test AVANT échantillonnage
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["project"]["session_id"]
    )

    # 2. Application de ta stratégie d'équilibrage sur le TRAIN uniquement
    initial_counts = Counter(y_train)
    target_size = 35346

    over_strat = {k: target_size for k, v in initial_counts.items() if v < target_size}
    under_strat = {k: target_size for k, v in initial_counts.items() if v > target_size}

    over = RandomOverSampler(sampling_strategy=over_strat, random_state=42)
    under = RandomUnderSampler(sampling_strategy=under_strat, random_state=42)

    pipeline_resample = ImbPipeline(steps=[("o", over), ("u", under)])
    X_train_res, y_train_res = pipeline_resample.fit_resample(X_train, y_train)

    logging.info(f"Distribution après équilibrage : {Counter(y_train_res)}")

    # 3. Recomposition des DataFrames pour sauvegarde Parquet
    train_resampled = pd.concat([X_train_res, y_train_res], axis=1)
    test_final = pd.concat([X_test, y_test], axis=1)

    # Sauvegarde
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    train_resampled.to_parquet(os.path.join(DATA_DIR, config["data"]["train_parquet"]), index=False)
    test_final.to_parquet(os.path.join(DATA_DIR, config["data"]["test_parquet"]), index=False)

    logging.info("Données équilibrées et sauvegardées en Parquet.")


if __name__ == "__main__":
    preprocess()
