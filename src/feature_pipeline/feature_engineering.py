# src/feature_pipeline/feature_engineering.py

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

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .engineered_features import (
    CHANNEL_METRIC_GROUPS,
    DIGITAL_SPEND_COLUMNS,
    EXTERNAL_METRIC_COLUMNS,
    OFFLINE_SPEND_COLUMNS,
)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a source column to numeric; invalid values become NaN."""

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide safely; zero or missing denominators return NaN."""

    numerator_numeric = clean_numeric_series(numerator)
    denominator_numeric = clean_numeric_series(
        denominator
    ).replace(0, np.nan)

    return (
        numerator_numeric.div(denominator_numeric)
        .replace([np.inf, -np.inf], np.nan)
    )


def sum_existing_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    """
    Sum configured columns that exist.

    Returns NaN when no configured source column exists, avoiding
    accidental interpretation of missing data as genuine zero spend.
    """

    existing_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing_columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    numeric_columns = pd.DataFrame(
        {
            column: clean_numeric_series(df[column])
            for column in existing_columns
        },
        index=df.index,
    )

    return numeric_columns.sum(axis=1, min_count=1)


def add_ratio_feature(
    df: pd.DataFrame,
    *,
    numerator_column: str | None,
    denominator_column: str | None,
    output_column: str,
) -> None:
    """Add a ratio only when both configured source columns exist."""

    if not numerator_column or not denominator_column:
        return

    if numerator_column not in df.columns:
        return

    if denominator_column not in df.columns:
        return

    df[output_column] = safe_divide(
        df[numerator_column],
        df[denominator_column],
    )


def add_channel_efficiency_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add:
    qs_cvr, ctr, cpi, cpc, cpqs, impression_share,
    top_impression_share and pct_top_impressions.
    """

    output = df.copy()

    for group_name, columns in CHANNEL_METRIC_GROUPS.items():
        sessions = columns.get("sessions")
        qs = columns.get("qs")
        clicks = columns.get("clicks")
        impressions = columns.get("impressions")
        cost = columns.get("cost")
        market_impressions = columns.get("market_impressions")
        top_impressions = columns.get("top_impressions")

        ratio_definitions = [
            (qs, sessions, f"{group_name}_qs_cvr"),
            (clicks, impressions, f"{group_name}_ctr"),
            (cost, impressions, f"{group_name}_cpi"),
            (cost, clicks, f"{group_name}_cpc"),
            (cost, qs, f"{group_name}_cpqs"),
            (
                impressions,
                market_impressions,
                f"{group_name}_impression_share",
            ),
            (
                top_impressions,
                market_impressions,
                f"{group_name}_top_impression_share",
            ),
            (
                top_impressions,
                impressions,
                f"{group_name}_pct_top_impressions",
            ),
        ]

        for numerator, denominator, output_column in ratio_definitions:
            add_ratio_feature(
                output,
                numerator_column=numerator,
                denominator_column=denominator,
                output_column=output_column,
            )

    return output


def add_spend_mix_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add engineered spend totals and spend-allocation percentages.
    """

    output = df.copy()

    digital_channel_spend: dict[str, pd.Series] = {}
    offline_channel_spend: dict[str, pd.Series] = {}

    for channel_name, columns in DIGITAL_SPEND_COLUMNS.items():
        spend_series = sum_existing_columns(output, columns)
        digital_channel_spend[channel_name] = spend_series
        output[
            f"engineered_spend_{channel_name}"
        ] = spend_series

    for channel_name, columns in OFFLINE_SPEND_COLUMNS.items():
        spend_series = sum_existing_columns(output, columns)
        offline_channel_spend[channel_name] = spend_series
        output[
            f"engineered_spend_{channel_name}"
        ] = spend_series

    digital_spend_frame = pd.DataFrame(
        digital_channel_spend,
        index=output.index,
    )

    offline_spend_frame = pd.DataFrame(
        offline_channel_spend,
        index=output.index,
    )

    output["engineered_spend_digital"] = (
        digital_spend_frame.sum(axis=1, min_count=1)
        if not digital_spend_frame.empty
        else pd.Series(np.nan, index=output.index)
    )

    output["engineered_spend_offline"] = (
        offline_spend_frame.sum(axis=1, min_count=1)
        if not offline_spend_frame.empty
        else pd.Series(np.nan, index=output.index)
    )

    output["engineered_spend_total"] = pd.concat(
        [
            output["engineered_spend_digital"],
            output["engineered_spend_offline"],
        ],
        axis=1,
    ).sum(axis=1, min_count=1)

    for channel_name, spend_series in digital_channel_spend.items():
        output[
            f"spend_{channel_name}_pct_digital"
        ] = safe_divide(
            spend_series,
            output["engineered_spend_digital"],
        )

        output[
            f"spend_{channel_name}_pct_total"
        ] = safe_divide(
            spend_series,
            output["engineered_spend_total"],
        )

    output["spend_digital_pct_total"] = safe_divide(
        output["engineered_spend_digital"],
        output["engineered_spend_total"],
    )

    output["spend_offline_pct_total"] = safe_divide(
        output["engineered_spend_offline"],
        output["engineered_spend_total"],
    )

    return output


def add_external_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add overall market-level impression features.
    """

    output = df.copy()

    impressions_col = EXTERNAL_METRIC_COLUMNS.get(
        "impressions"
    )
    market_impressions_col = EXTERNAL_METRIC_COLUMNS.get(
        "market_impressions"
    )
    top_impressions_col = EXTERNAL_METRIC_COLUMNS.get(
        "top_impressions"
    )

    print("\n==external market feature inputs")
    print(
        f"{impressions_col}: "
        f"{impressions_col in output.columns}"
    )
    print(
        f"{market_impressions_col}: "
        f"{market_impressions_col in output.columns}"
    )
    print(
        f"{top_impressions_col}: "
        f"{top_impressions_col in output.columns}"
    )

    if (
        impressions_col in output.columns
        and market_impressions_col in output.columns
    ):
        output["overall_impression_share"] = safe_divide(
            output[impressions_col],
            output[market_impressions_col],
        )

    if (
        top_impressions_col in output.columns
        and impressions_col in output.columns
    ):
        output["overall_top_impression_rate"] = safe_divide(
            output[top_impressions_col],
            output[impressions_col],
        )

    return output

def add_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = df.copy()

    output = add_channel_efficiency_features(
        output
    )

    output = add_spend_mix_features(
        output
    )

    output = add_external_market_features(
        output
    )

    return output
