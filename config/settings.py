"""
Global project settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ENGINEERED_DIR = PROJECT_ROOT / "data" / "engineered"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

DATA_SOURCE = "csv"
# DATA_SOURCE = "bigquery"

DATE_COLUMN = "date_day"
RANDOM_STATE = 42
