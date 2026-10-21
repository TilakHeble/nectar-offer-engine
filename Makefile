.PHONY: install ingest dbt dbt-docs eda test lint all clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev]"

ingest:
	python -m nectar.ingest.load_raw --raw-dir $(RAW_DIR)

dbt:
	cd dbt && dbt build --profiles-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_segments_and_funnel.ipynb

test:
	pytest -q

lint:
	ruff check src tests

all: ingest dbt test eda

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb
