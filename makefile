.PHONY: train evaluate all

train:
	python3 scripts/run_train.py

evaluate:
	python3 scripts/run_evaluate.py

all: train evaluate
