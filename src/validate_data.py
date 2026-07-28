"""Validation utilities for the Marketing Optimisation Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

DATE_CANDIDATES = ("date_day", "event_date", "qs_date")


@dataclass
class ValidationResult:
    dataset: str
    status: str
    rows: int
    columns: int
    date_column: str | None
    duplicate_rows: int
    duplicate_keys: int
    null_dates: int
    min_date: str | None
    max_date: str | None
    issues: str


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = (
        output.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return output


def find_date_column(df: pd.DataFrame) -> str | None:
    for column in DATE_CANDIDATES:
        if column in df.columns:
            return column
    return None


def parse_date_column(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    output = df.copy()
    output[date_column] = pd.to_datetime(
        output[date_column],
        errors="coerce",
    ).dt.normalize()
    return output


def infer_key_columns(dataset_name: str, df: pd.DataFrame, date_column: str) -> list[str]:
    key_map = {
        "data_dates": [date_column],
        "data_funnel_uncohorted": [date_column],
        "data_funnel_cohorted": [date_column],
        "data_media_inputs": [date_column, "channel"],
        "media_input": [date_column, "channel"],
        "media_inputs": [date_column, "channel"],
        "data_marketing_responses": [date_column, "metric"],
        "marketing_responses": [date_column, "metric"],
        "data_attribution": [date_column, "channel"],
        "attribution": [date_column, "channel"],
        "data_external": [date_column, "metric"],
        "external": [date_column, "metric"],
    }
    return [column for column in key_map.get(dataset_name, [date_column]) if column in df.columns]


def validate_dataset(path: str | Path) -> ValidationResult:
    path = Path(path)
    dataset_name = path.stem
    issues: list[str] = []

    df = normalise_columns(pd.read_csv(path))
    date_column = find_date_column(df)

    if date_column is None:
        return ValidationResult(
            dataset_name, "FAIL", len(df), len(df.columns), None,
            int(df.duplicated().sum()), 0, 0, None, None,
            "no recognised date column",
        )

    df = parse_date_column(df, date_column)
    null_dates = int(df[date_column].isna().sum())
    duplicate_rows = int(df.duplicated().sum())
    keys = infer_key_columns(dataset_name, df, date_column)
    duplicate_keys = int(df.duplicated(subset=keys).sum()) if keys else 0

    if null_dates:
        issues.append(f"{null_dates} invalid or null dates")
    if duplicate_rows:
        issues.append(f"{duplicate_rows} fully duplicated rows")
    if duplicate_keys:
        issues.append(f"{duplicate_keys} duplicate rows at grain {keys}")

    future_dates = int((df[date_column] > pd.Timestamp.today().normalize()).sum())
    if future_dates:
        issues.append(f"{future_dates} future-dated rows")

    required_by_dataset = {
        "data_media_inputs": ["channel"],
        "media_input": ["channel"],
        "media_inputs": ["channel"],
        "data_marketing_responses": ["metric", "value"],
        "marketing_responses": ["metric", "value"],
        "data_external": ["metric", "value"],
        "external": ["metric", "value"],
    }
    missing = [c for c in required_by_dataset.get(dataset_name, []) if c not in df.columns]
    if missing:
        issues.append("missing required columns: " + ", ".join(missing))

    min_date = df[date_column].min()
    max_date = df[date_column].max()

    return ValidationResult(
        dataset=dataset_name,
        status="PASS" if not issues else "FAIL",
        rows=len(df),
        columns=len(df.columns),
        date_column=date_column,
        duplicate_rows=duplicate_rows,
        duplicate_keys=duplicate_keys,
        null_dates=null_dates,
        min_date=min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
        max_date=max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
        issues="; ".join(issues),
    )


def validate_raw_folder(
    raw_path: str | Path = "data/raw",
    output_path: str | Path = "data/outputs/validation_report.csv",
) -> pd.DataFrame:
    raw_path = Path(raw_path)
    csv_paths = sorted(raw_path.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    report = pd.DataFrame([asdict(validate_dataset(path)) for path in csv_paths])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def assert_validation_passed(report: pd.DataFrame) -> None:
    failed = report.loc[report["status"] != "PASS"]
    if not failed.empty:
        details = failed[["dataset", "issues"]].to_string(index=False)
        raise ValueError(f"Raw-data validation failed:\n\n{details}")
