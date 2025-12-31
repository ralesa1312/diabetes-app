# Makefile
.PHONY: install test run_pipeline mlflow clean

install:
	pip install --upgrade pip
	# Installe le projet en mode éditable avec les dépendances
	pip install -e .

test:
	pytest tests/ -v --cov=ml

run_pipeline:
	@echo "Lancement du pipeline MLOps..."
	python -m ml.preprocess
	python -m ml.train
	python -m ml.evaluate

mlflow:
	@echo "Lancement de l'interface MLflow sur http://localhost:5000"
	mlflow ui --port 5000

docker-build:
	docker build -t diabetes-mlops-pipeline .

docker-run:
	# Note: On monte les volumes pour récupérer les fichiers produits sur l'hôte
	docker run --rm \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/models:/app/models \
		diabetes-mlops-pipeline

clean:
	rm -rf .pytest_cache
	rm -rf mlruns/
	rm *.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +