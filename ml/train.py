import pandas as pd
import yaml
import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import DATA_DIR, MODEL_DIR, save_model

logging.basicConfig(level=logging.INFO)

def train():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet(os.path.join(DATA_DIR, config['data']['train_parquet']))
    X = train_df.drop(columns=[config['data']['target']])
    y = train_df[config['data']['target']]

    models_to_test = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier()
    }

    best_auc = 0
    best_pipeline = None
    best_name = ""

    mlflow.set_experiment(config['model']['experiment_name'])

    for name, model in models_to_test.items():
        with mlflow.start_run(run_name=name, nested=True):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])

            # cross_validate permet de calculer plusieurs métriques d'un coup
            # 'roc_auc_ovr' est utilisé car votre cible peut avoir 3 classes (0,1,2)
            cv_results = cross_validate(
                pipeline, X, y, cv=5, 
                scoring=['accuracy', 'roc_auc_ovr'],
                return_train_score=False
            )

            mean_accuracy = cv_results['test_accuracy'].mean()
            mean_auc = cv_results['test_roc_auc_ovr'].mean()

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", mean_accuracy)
            mlflow.log_metric("auc", mean_auc)
            
            logging.info(f"{name}: Accuracy={mean_accuracy:.4f}, AUC={mean_auc:.4f}")

            # On choisit le meilleur modèle sur la base de l'AUC
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_pipeline = pipeline
                best_name = name

    # Finalisation
    logging.info(f"--- Meilleur Modèle (basé sur AUC) : {best_name} ---")
    best_pipeline.fit(X, y)
    
    with mlflow.start_run(run_name="FINAL_CHOICE"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("final_auc", best_auc)
        mlflow.sklearn.log_model(best_pipeline, "model")
        
        save_model(best_pipeline, os.path.join(MODEL_DIR, f"{config['model']['name']}.pkl"))

if __name__ == "__main__":
    train()