.PHONY: install ingest dbt dbt-docs simulate train optimise mlflow-ui eda test lint all clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev,ml,serve]"

ingest:
	python -m nectar.ingest.load_raw --raw-dir $(RAW_DIR)

dbt:
	cd dbt && dbt build --profiles-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .

simulate:
	python -m nectar.simulate.simulator

train:
	python -m nectar.models.train

optimise:
	python -m nectar.optimise.run --budget 150 --max-per-member 99

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

bundle:
	python -m nectar.serving.bundle --budget 600 --max-per-member 3

serve:
	uvicorn nectar.serving.app:app --reload

docker-build:
	docker build -t nectar-offer-api:latest .

docker-run:
	docker compose up -d

smoke:
	python -m nectar.serving.smoke --url http://localhost:8000

benchmark:
	python -m nectar.serving.benchmark --url http://localhost:8000

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_segments_and_funnel.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_offer_experiment.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/04_uplift_models.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/05_offer_allocation.ipynb

test:
	pytest -q

lint:
	ruff check src tests

all: ingest dbt simulate train optimise bundle test eda

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb
