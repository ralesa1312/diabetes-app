# 1. Image de base
FROM python:3.11-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier les fichiers de configuration 
COPY pyproject.toml .

COPY README.md* . 

# 5. Installer les dépendances listées dans pyproject.toml

RUN pip install --no-cache-dir .

# 6. Copier le reste du code
COPY . .

# 7. Créer les dossiers de données
RUN mkdir -p data/raw data/preprocessed models

# 8. Commande de démarrage
CMD ["make", "run_pipeline"]