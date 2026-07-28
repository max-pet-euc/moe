"""Validation checks for raw and engineered data."""

import pandas as pd


def validate_date_column(df: pd.DataFrame, date_column: str = "date_day") -> None:
    if date_column not in df.columns:
        raise ValueError(f"Missing required date column: {date_column}")

    if df[date_column].isna().any():
        raise ValueError(f"Null values found in {date_column}")
