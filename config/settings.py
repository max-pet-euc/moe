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

#=========================================================
#==granularity

MODEL_GRAIN = "daily"
# MODEL_GRAIN = "weekly"
# MODEL_GRAIN = "monthly"

VALID_MODEL_GRAINS = {
    "daily",
    "weekly",
    "monthly",
}

MODEL_GRAIN_COLUMNS = {
    "daily": "date_day",
    "weekly": "date_week",
    "monthly": "date_month",
}

#=========================================================
#==modelling config

MODEL_STAGE = "cp"

TEST_DAYS = 90

RANDOM_STATE = 42

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

STAGE_ALIASES = {
    "us": ["us"],
    "qs": ["qs"],
    "qc": ["qc"],
    "qe": ["qe"],
    "cp": ["cp"],
    "op": ["op"],
    "rv": [
        "rv",
        "revenue",
    ],
}

ALWAYS_EXCLUDED_COLUMNS = {
    "date_day",
    "date_week",
    "date_month",
    "spend_total",
    "spend_media",
    "spend_digital",
    "uncohorted_us",
    "uncohorted_qs",
    "uncohorted_qc",
    "uncohorted_qe",
    "cohorted_us",
    "cohorted_qs",
    "cohorted_qc",
    "cohorted_qe",
}

MODEL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "models"
)

PREDICTION_OUTPUT_DIR = (
    OUTPUT_DIR
    / "predictions"
)

REPORT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "reports"
)
