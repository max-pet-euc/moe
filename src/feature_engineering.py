"""
Build the canonical daily MOE feature dataset.

Launched by:
    notebooks/01_feature_builder.ipynb

Purpose:
    - Load validated raw datasets
    - Reshape and merge all datasets to a daily grain
    - Generate engineered features (lags, rolling averages, transforms, etc.)
    - Save data/engineered/data_features.csv

Output:
    data/engineered/data_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validate_data import find_date_column, normalise_columns, parse_date_column

FILE_ALIASES = {
    "dates": ["data_dates.csv"],
    "funnel_uncohorted": ["data_funnel_uncohorted.csv"],
    "funnel_cohorted": ["data_funnel_cohorted.csv"],
    "media_inputs": ["data_media_inputs.csv", "media_inputs.csv", "media_input.csv"],
    "marketing_responses": ["data_marketing_responses.csv", "marketing_responses.csv", "marketing_response.csv"],
    "attribution": ["data_attribution.csv", "attribution.csv"],
    "external": ["data_external.csv", "external.csv"],
}


def find_file(raw_path: str | Path, aliases: list[str]) -> Path | None:
    raw_path = Path(raw_path)
    for filename in aliases:
        path = raw_path / filename
        if path.exists():
            return path
    return None


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    df = normalise_columns(pd.read_csv(path))
    date_column = find_date_column(df)
    if date_column is None:
        raise ValueError(f"No recognised date column in {path}")
    df = parse_date_column(df, date_column)
    if date_column != "date_day":
        df = df.rename(columns={date_column: "date_day"})
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def pivot_channel_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    channel_column = "channel" if "channel" in df.columns else "platform" if "platform" in df.columns else None
    if channel_column is None:
        raise ValueError(f"{prefix} requires channel or platform")

    output = df.copy()
    metric_columns = [c for c in output.columns if c not in {"date_day", channel_column}]
    for column in metric_columns:
        output[column] = clean_numeric(output[column])

    output = output.pivot_table(
        index="date_day",
        columns=channel_column,
        values=metric_columns,
        aggfunc="sum",
        fill_value=0,
    )
    output.columns = [f"{prefix}_{str(channel).lower()}_{metric}" for metric, channel in output.columns]
    return output.reset_index()


def pivot_metric_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"date_day", "metric", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"{prefix} requires columns: {sorted(required)}")

    output = df.copy()
    output["value"] = clean_numeric(output["value"])
    output["metric"] = (
        output["metric"].astype(str)
        .str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    output = output.pivot_table(index="date_day", columns="metric", values="value", aggfunc="sum")
    output.columns = [f"{prefix}_{column}" for column in output.columns]
    return output.reset_index()


def prepare_daily_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = df.copy()
    rename_map = {column: f"{prefix}_{column}" for column in output.columns if column != "date_day"}
    return output.rename(columns=rename_map)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.sort_values("date_day").copy()
    protected = ("funnel_uncohorted_", "funnel_cohorted_")
    numeric_columns = [
        c for c in output.select_dtypes(include="number").columns
        if not c.startswith(protected)
    ]

    additions = {}
    for column in numeric_columns:
        values = output[column]
        additions[f"{column}_lag_1"] = values.shift(1)
        additions[f"{column}_lag_7"] = values.shift(7)
        additions[f"{column}_roll_7"] = values.shift(1).rolling(7).mean()
        additions[f"{column}_roll_28"] = values.shift(1).rolling(28).mean()
        if (values.dropna() >= 0).all():
            additions[f"{column}_log1p"] = np.log1p(values)

    if additions:
        output = pd.concat([output, pd.DataFrame(additions)], axis=1)
    return output


def build_features(
    raw_path: str | Path = "data/raw",
    output_path: str | Path = "data/engineered/data_features.csv",
) -> pd.DataFrame:
    raw_path = Path(raw_path)
    found = {name: find_file(raw_path, aliases) for name, aliases in FILE_ALIASES.items()}

    if found["dates"] is None:
        raise FileNotFoundError("data_dates.csv is required")

    features = load_raw_csv(found["dates"]).drop_duplicates("date_day")

    builders = {
        "funnel_uncohorted": lambda df: prepare_daily_table(df, "funnel_uncohorted"),
        "funnel_cohorted": lambda df: prepare_daily_table(df, "funnel_cohorted"),
        "media_inputs": lambda df: pivot_channel_table(df, "media"),
        "marketing_responses": lambda df: pivot_metric_table(df, "response"),
        "attribution": lambda df: pivot_metric_table(df, "attribution") if {"metric", "value"}.issubset(df.columns) else pivot_channel_table(df, "attribution"),
        "external": lambda df: pivot_metric_table(df, "external") if {"metric", "value"}.issubset(df.columns) else prepare_daily_table(df, "external"),
    }

    for name, builder in builders.items():
        path = found[name]
        if path is None:
            print(f"Skipping {name}: file not found")
            continue
        table = builder(load_raw_csv(path))
        features = features.merge(table, on="date_day", how="left", validate="one_to_one")

    features = add_basic_features(features).sort_values("date_day")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features
