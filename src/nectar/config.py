"""Project paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "nectar.duckdb"
FIGURES_DIR = ROOT / "reports" / "figures"

# The Complete Journey covers ~2 years of shopping, expressed as day numbers 1..711.
N_DAYS = 711
