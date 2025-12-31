Diabetes Classification Pipeline (MLOps)

Ce projet implémente un pipeline de Machine Learning industriel pour la classification des indicateurs de santé liés au diabète. Il intègre les meilleures pratiques MLOps : automatisation, traçabilité, tests unitaires et conteneurisation.
    Fonctionnalités

    Data Engineering : Nettoyage et équilibrage des classes via imblearn (Over/Under sampling).

    Format Parquet : Utilisation du stockage colonnaire pour l'efficacité des données.

    Model Tracking : Comparaison de plusieurs modèles (Random Forest, Gradient Boosting, etc.) avec MLflow.

    Metrics : Évaluation basée sur l'AUC et la Matrice de Confusion.

    Qualité : Suite de tests unitaires avec pytest.

    Portabilité : Entièrement conteneurisé avec Docker.

    Installation & Setup
Prérequis

    Python 3.11+

    Make (facultatif mais recommandé)

    Docker (pour la version conteneurisée)

Installation locale
Bash

# Installer les dépendances via pyproject.toml
make install

    Utilisation
1. Exécuter le pipeline complet

Cette commande enchaîne le prétraitement, l'entraînement et l'évaluation :
Bash

make run_pipeline

2. Suivre les expériences (MLflow)

Pour visualiser les performances et comparer les modèles :
Bash

make mlflow

Puis ouvrez http://localhost:5000 dans votre navigateur.
3. Lancer les tests
Bash

make test

    Docker

Pour garantir une reproductibilité totale sans installer Python localement :
Bash

# Construire l'image
make docker-build

# Lancer le pipeline dans le conteneur
make docker-run

 Structure du Projet
Plaintext

├── ml/                 # Cœur du code (Scripts de traitement et d'entraînement)
├── data/
│   ├── raw/            # Données sources (CSV)
│   └── preprocessed/   # Données transformées (Parquet)
├── models/             # Artefacts (.pkl) et graphiques d'évaluation
├── tests/              # Tests unitaires et d'intégration
├── config.yaml         # Paramètres de configuration
├── pyproject.toml      # Gestion des dépendances
└── Makefile            # Automatisation des tâches

    Résultats Attendus

Après l'exécution, vous trouverez dans le dossier models/ :

    best_diabetes_model.pkl : Le pipeline de prédiction final.

    evaluation_metrics.png : Visualisation des performances (Matrice de confusion).