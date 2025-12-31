import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .utils import DATA_DIR, MODEL_DIR, load_model


def evaluate():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = load_model(os.path.join(MODEL_DIR, f"{config['model']['name']}.pkl"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, config["data"]["test_parquet"]))

    X_test = test_df.drop(columns=[config["data"]["target"]])
    y_test = test_df[config["data"]["target"]]

    # 1. Prédictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)  # Nécessaire pour l'AUC

    # 2. Calcul de l'AUC (stratégie One-vs-Rest pour multiclasse)
    auc_score = roc_auc_score(y_test, y_probs, multi_class="ovr")
    print(f"\nScore AUC final sur Test Set : {auc_score:.4f}")

    # 3. Rapport de Classification
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))

    # 4. Visualisation (Matrice de Confusion)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Matrice de Confusion")

    # 5. Visualisation (Note: La courbe ROC multiclasse est complexe, ici on affiche l'essentiel)
    plt.subplot(1, 2, 2)
    plt.text(0.3, 0.5, f"AUC Score: {auc_score:.4f}", fontsize=15)
    plt.title("Métrique AUC")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "metrics/evaluation_metrics.png"))
    print(f"Graphiques sauvegardés dans {DATA_DIR}")


if __name__ == "__main__":
    evaluate()
