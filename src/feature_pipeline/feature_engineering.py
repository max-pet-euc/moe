# src/feature_pipeline/feature_engineering.py
# FEATURE_ENGINEERING_VERSION = "2.1-20260730_1807"

"""
Generate engineered features for the canonical daily MOE dataset.

Channel efficiency metrics are calculated using matching channel-level
metrics from data_digital_platforms.csv.

Generated channel metrics include:
- QS conversion rate
- CP conversion rate
- CTR
- CPC
- CPM
- cost per impression
- cost per QS
- cost per CP
- impression-share measures
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


def clean_numeric_series(
    series: pd.Series,
) -> pd.Series:
    """Convert a source column to numeric; invalid values become NaN."""

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    """Divide safely; zero or missing denominators return NaN."""

    numerator_numeric = clean_numeric_series(
        numerator
    )

    denominator_numeric = clean_numeric_series(
        denominator
    ).replace(
        0,
        np.nan,
    )

    return (
        numerator_numeric
        .mul(multiplier)
        .div(denominator_numeric)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def sum_existing_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    """
    Sum configured columns that exist.

    Returns NaN when no configured source column exists.
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
            column: clean_numeric_series(
                df[column]
            )
            for column in existing_columns
        },
        index=df.index,
    )

    return numeric_columns.sum(
        axis=1,
        min_count=1,
    )


def add_ratio_feature(
    df: pd.DataFrame,
    *,
    numerator_column: str | None,
    denominator_column: str | None,
    output_column: str,
    multiplier: float = 1.0,
) -> bool:
    """
    Add a ratio only when both source columns exist.

    Returns True when the feature was created.
    """

    if not numerator_column or not denominator_column:
        return False

    if numerator_column not in df.columns:
        return False

    if denominator_column not in df.columns:
        return False

    df[output_column] = safe_divide(
        df[numerator_column],
        df[denominator_column],
        multiplier=multiplier,
    )

    return True


def add_channel_efficiency_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add channel-level relative and efficiency metrics.

    Digital spend is sourced from the matching channel row in
    data_digital_platforms.csv.
    """

    output = df.copy()
    created_features: list[str] = []

    for group_name, columns in CHANNEL_METRIC_GROUPS.items():
        sessions = columns.get("sessions")
        qs = columns.get("qs")
        cp = columns.get("cp")
        clicks = columns.get("clicks")
        impressions = columns.get("impressions")
        cost = columns.get("cost")
        market_impressions = columns.get(
            "market_impressions"
        )
        top_impressions = columns.get(
            "top_impressions"
        )

        ratio_definitions = [
            (qs, sessions, f"{group_name}_qs_cvr", 1.0),
            (cp, qs, f"{group_name}_cp_cvr", 1.0),
            (clicks, impressions, f"{group_name}_ctr", 1.0),
            (cost, impressions, f"{group_name}_cpi", 1.0),
            (cost, impressions, f"{group_name}_cpm", 1000.0),
            (cost, clicks, f"{group_name}_cpc", 1.0),
            (cost, qs, f"{group_name}_cpqs", 1.0),
            (cost, cp, f"{group_name}_cpcp", 1.0),
            (
                impressions,
                market_impressions,
                f"{group_name}_impression_share",
                1.0,
            ),
            (
                top_impressions,
                market_impressions,
                f"{group_name}_top_impression_share",
                1.0,
            ),
            (
                top_impressions,
                impressions,
                f"{group_name}_pct_top_impressions",
                1.0,
            ),
        ]

        for (
            numerator,
            denominator,
            output_column,
            multiplier,
        ) in ratio_definitions:
            created = add_ratio_feature(
                output,
                numerator_column=numerator,
                denominator_column=denominator,
                output_column=output_column,
                multiplier=multiplier,
            )

            if created:
                created_features.append(
                    output_column
                )

    print("\n==channel efficiency features")
    print(f"created: {len(created_features)}")

    for feature_name in created_features:
        populated_rows = int(
            output[feature_name]
            .notna()
            .sum()
        )

        print(
            f"{feature_name}: "
            f"{populated_rows:,} populated rows"
        )

    return output


def add_spend_mix_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add engineered spend totals and spend-allocation percentages.

    These totals use data_media_inputs.csv and remain separate from
    detailed channel efficiency features.
    """

    output = df.copy()

    digital_channel_spend: dict[
        str,
        pd.Series,
    ] = {}

    offline_channel_spend: dict[
        str,
        pd.Series,
    ] = {}

    for channel_name, columns in DIGITAL_SPEND_COLUMNS.items():
        spend_series = sum_existing_columns(
            output,
            columns,
        )

        digital_channel_spend[
            channel_name
        ] = spend_series

        output[
            f"engineered_spend_{channel_name}"
        ] = spend_series

    for channel_name, columns in OFFLINE_SPEND_COLUMNS.items():
        spend_series = sum_existing_columns(
            output,
            columns,
        )

        offline_channel_spend[
            channel_name
        ] = spend_series

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
        digital_spend_frame.sum(
            axis=1,
            min_count=1,
        )
        if not digital_spend_frame.empty
        else pd.Series(
            np.nan,
            index=output.index,
        )
    )

    output["engineered_spend_offline"] = (
        offline_spend_frame.sum(
            axis=1,
            min_count=1,
        )
        if not offline_spend_frame.empty
        else pd.Series(
            np.nan,
            index=output.index,
        )
    )

    output["engineered_spend_total"] = (
        pd.concat(
            [
                output[
                    "engineered_spend_digital"
                ],
                output[
                    "engineered_spend_offline"
                ],
            ],
            axis=1,
        )
        .sum(
            axis=1,
            min_count=1,
        )
    )

    for channel_name, spend_series in digital_channel_spend.items():
        output[
            f"spend_{channel_name}_pct_digital"
        ] = safe_divide(
            spend_series,
            output[
                "engineered_spend_digital"
            ],
        )

        output[
            f"spend_{channel_name}_pct_total"
        ] = safe_divide(
            spend_series,
            output[
                "engineered_spend_total"
            ],
        )

    output["spend_digital_pct_total"] = (
        safe_divide(
            output[
                "engineered_spend_digital"
            ],
            output[
                "engineered_spend_total"
            ],
        )
    )

    output["spend_offline_pct_total"] = (
        safe_divide(
            output[
                "engineered_spend_offline"
            ],
            output[
                "engineered_spend_total"
            ],
        )
    )

    return output


def add_external_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add overall market-level impression features."""

    output = df.copy()

    impressions_col = (
        EXTERNAL_METRIC_COLUMNS.get(
            "impressions"
        )
    )

    market_impressions_col = (
        EXTERNAL_METRIC_COLUMNS.get(
            "market_impressions"
        )
    )

    top_impressions_col = (
        EXTERNAL_METRIC_COLUMNS.get(
            "top_impressions"
        )
    )

    if (
        impressions_col in output.columns
        and market_impressions_col in output.columns
    ):
        output[
            "overall_impression_share"
        ] = safe_divide(
            output[impressions_col],
            output[market_impressions_col],
        )

    if (
        top_impressions_col in output.columns
        and impressions_col in output.columns
    ):
        output[
            "overall_top_impression_rate"
        ] = safe_divide(
            output[top_impressions_col],
            output[impressions_col],
        )

    return output


def add_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add all configured engineered features."""

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