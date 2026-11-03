.PHONY: test train lint clean

train:
	python train.py

test:
	pytest -v

lint:
	flake8 risk_engine

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache