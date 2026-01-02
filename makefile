# Variables
IMAGE_NAME = diabetes-mlops-app
CONTAINER_PORT = 8501

.PHONY: install test run_pipeline streamlit mlflow docker-build docker-run clean

install:
	pip install --upgrade pip
	pip install -e .

test:
	pytest tests/ -v --cov=ml

# Pipeline d'entraînement (MLOps)
run_pipeline:
	@echo " Lancement du pipeline d'entraînement..."
	python3 -m ml.preprocess
	python3 -m ml.train
	python3 -m ml.evaluate

# Lancement de l'interface utilisateur
streamlit:
	@echo " Lancement de l'Assistant Diabète sur http://localhost:8501"
	streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0

mlflow:
	@echo "Lancement de l'interface MLflow sur http://localhost:5000"
	mlflow ui --port 5000

# --- DOCKER ---

docker-build:
	@echo " Construction de l'image Docker..."
	docker build -t $(IMAGE_NAME) .

docker-run:
	@echo " Lancement du conteneur (App + ML)..."
	# On expose le port 8501 et on monte les volumes pour la persistance
	docker run --rm -p $(CONTAINER_PORT):8501 \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/models:/app/models \
		$(IMAGE_NAME)

clean:
	@echo " Nettoyage des fichiers temporaires..."
	rm -rf .pytest_cache
	rm -rf mlruns/
	rm -f *.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +