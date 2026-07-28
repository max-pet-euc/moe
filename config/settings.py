"""
Global project settings.
"""

from pathlib import Path


# ==========================================================
# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
ENGINEERED_DIR = DATA_DIR / "engineered"
OUTPUT_DIR = DATA_DIR / "outputs"


# ==========================================================
# Data source

DATA_SOURCE = "csv"
# DATA_SOURCE = "bigquery"


# ==========================================================
# Modelling

DATE_COLUMN = "date_day"

RANDOM_STATE = 42

TEST_DAYS = 90


# ==========================================================
# Funnel

FUNNEL_ORDER = [
    "us",
    "qs",
    "qc",
    "qe",
    "cp",
    "op",
    "rv",
]

MODEL_TARGETS = {
    "qs": "uncohorted_qs",
    "cp": "uncohorted_cp",
    "rv": "uncohorted_rv",
}


# ==========================================================
# Always excluded features

ALWAYS_EXCLUDED_COLUMNS = {
    DATE_COLUMN,
}
