# Makefile
.PHONY: install test run_pipeline mlflow clean

install:
	pip install -r requirements.txt

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

clean:
	rm -rf .pytest_cache
	rm -rf mlruns/
	find . -type d -name "__pycache__" -exec rm -rf {} +