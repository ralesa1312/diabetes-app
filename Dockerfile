# 1. Image de base légère
FROM python:3.11-slim

# 2. Variables d'environnement
# PYTHONUNBUFFERED force l'affichage des logs en temps réel
# PYTHONPATH permet de trouver le module 'ml' depuis n'importe où
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Dossier de travail
WORKDIR /app

# 4. Dépendances système
# Ajout de 'make' explicitement pour être sûr que les commandes Makefile fonctionnent
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

# 5. Copier les fichiers de définition du projet
COPY pyproject.toml .
COPY README.md* . 

# 6. Installer les dépendances
# On utilise -e . pour que les imports 'from ml.utils' fonctionnent partout
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 7. Copier l'intégralité du projet
COPY . .

# 8. Création des répertoires de données
RUN mkdir -p data/raw data/preprocessed models

# 9. Exposer le port Streamlit
EXPOSE 8501

# 10. Commande de démarrage
# On lance Streamlit par défaut. 
# Note: L'adresse 0.0.0.0 est obligatoire pour Docker
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]