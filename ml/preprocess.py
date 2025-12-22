# ml/preprocess.py

import logging
from collections import Counter
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


def preprocess_data(
    df: pd.DataFrame,
    target: str,
    target_size: int = 35346,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Nettoyage, équilibrage des classes (over + under sampling) et normalisation.

    Parameters
    ----------
    df : pd.DataFrame
        Jeu de données complet
    target : str
        Nom de la variable cible
    target_size : int, optional
        Taille cible par classe après équilibrage
    random_state : int, optional
        Graine aléatoire pour la reproductibilité

    Returns
    -------
    X_scaled : pd.DataFrame
        Features normalisées
    y : pd.Series
        Variable cible
    scaler : StandardScaler
        Scaler entraîné (à sauvegarder pour l'inférence)
    """

    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' n'existe pas dans le DataFrame")

    # Copie défensive
    df = df.copy()

    # Suppression des lignes avec valeurs manquantes
    df.dropna(inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    class_counts = Counter(y)
    logger.info(f"Distribution initiale des classes: {class_counts}")

    # Stratégies d'équilibrage
    oversampling_strategy = {
        cls: target_size for cls, count in class_counts.items() if count < target_size
    }

    undersampling_strategy = {
        cls: target_size for cls, count in class_counts.items() if count > target_size
    }

    logger.info(f"Oversampling strategy: {oversampling_strategy}")
    logger.info(f"Undersampling strategy: {undersampling_strategy}")

    steps = []

    if oversampling_strategy:
        steps.append(
            (
                "over",
                RandomOverSampler(
                    sampling_strategy=oversampling_strategy,
                    random_state=random_state,
                ),
            )
        )

    if undersampling_strategy:
        steps.append(
            (
                "under",
                RandomUnderSampler(
                    sampling_strategy=undersampling_strategy,
                    random_state=random_state,
                ),
            )
        )

    if steps:
        pipeline = Pipeline(steps=steps)
        X, y = pipeline.fit_resample(X, y)

    # Sécurité typage
    y = y.astype(int)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    logger.info(f"Distribution finale des classes: {Counter(y)}")

    return X_scaled, y, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    X_train_path: Optional[str] = None,
    X_test_path: Optional[str] = None,
    y_train_path: Optional[str] = None,
    y_test_path: Optional[str] = None,
):
    """
    Sépare les données en train/test et sauvegarde les fichiers si les chemins sont fournis.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # important pour classification
    )

    if X_train_path:
        X_train.to_csv(X_train_path, index=False)
    if X_test_path:
        X_test.to_csv(X_test_path, index=False)
    if y_train_path:
        y_train.to_csv(y_train_path, index=False)
    if y_test_path:
        y_test.to_csv(y_test_path, index=False)

    return X_train, X_test, y_train, y_test
