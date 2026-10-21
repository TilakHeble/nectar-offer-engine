.PHONY: install ingest eda test lint all clean

RAW_DIR ?= data/raw

install:
	pip install -e ".[dev]"

ingest:
	python -m nectar.ingest.load_raw --raw-dir $(RAW_DIR)

eda:
	python -m nectar.eda_report
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

test:
	pytest -q

lint:
	ruff check src tests

all: ingest test eda

clean:
	rm -rf data/processed/*.parquet data/warehouse/*.duckdb
