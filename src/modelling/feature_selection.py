"""
Select valid model features using stage-aware leakage rules.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import (
    ALWAYS_EXCLUDED_COLUMNS,
    FUNNEL_ORDER,
    MODEL_TARGETS,
)


STAGE_ALIASES = {
    "us": ["us"],
    "qs": ["qs"],
    "qc": ["qc"],
    "qe": ["qe"],
    "cp": ["cp"],
    "op": ["op"],
    "rv": ["rv", "revenue"],
}


@dataclass
class FeatureSelectionResult:
    x: pd.DataFrame
    y: pd.Series
    target_column: str
    feature_columns: list[str]
    leakage_stages: list[str]
    leakage_columns: list[str]
    excluded_columns: list[str]


def clean_numeric_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "£",
            "",
            regex=False,
        )
        .str.replace(
            "%",
            "",
            regex=False,
        )
        .str.strip(),
        errors="coerce",
    )


def get_target_column(
    model_stage: str,
) -> str:
    if model_stage not in MODEL_TARGETS:
        raise ValueError(
            f"model_stage must be one of: "
            f"{list(MODEL_TARGETS)}"
        )

    return MODEL_TARGETS[model_stage]


def get_leakage_stages(
    model_stage: str,
) -> list[str]:
    if model_stage not in FUNNEL_ORDER:
        raise ValueError(
            f"unknown model stage: {model_stage}"
        )

    target_index = FUNNEL_ORDER.index(
        model_stage
    )

    return FUNNEL_ORDER[target_index:]


def contains_stage_marker(
    column: str,
    stages: list[str],
) -> bool:
    column_lower = column.lower()

    markers = [
        alias
        for stage in stages
        for alias in STAGE_ALIASES[stage]
    ]

    return any(
        re.search(
            rf"(?:^|_){re.escape(marker)}(?:_|$)",
            column_lower,
        )
        is not None
        for marker in markers
    )


def select_features(
    model_df: pd.DataFrame,
    model_stage: str,
) -> FeatureSelectionResult:
    target_column = get_target_column(
        model_stage
    )

    if target_column not in model_df.columns:
        raise ValueError(
            f"target column not found: {target_column}"
        )

    leakage_stages = get_leakage_stages(
        model_stage
    )

    leakage_columns = [
        column
        for column in model_df.columns
        if column != target_column
        and contains_stage_marker(
            column=column,
            stages=leakage_stages,
        )
    ]

    candidate_columns = [
        column
        for column in model_df.columns
        if column not in ALWAYS_EXCLUDED_COLUMNS
        and column != target_column
        and column not in leakage_columns
    ]

    numeric_feature_data = {}
    excluded_columns = []

    for column in candidate_columns:
        cleaned_column = clean_numeric_series(
            model_df[column]
        )

        if cleaned_column.notna().any():
            numeric_feature_data[column] = (
                cleaned_column
            )
        else:
            excluded_columns.append(column)

    if not numeric_feature_data:
        raise ValueError(
            "no numeric feature columns found"
        )

    x = pd.DataFrame(
        numeric_feature_data,
        index=model_df.index,
    )

    x = (
        x
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    y = clean_numeric_series(
        model_df[target_column]
    )

    return FeatureSelectionResult(
        x=x,
        y=y,
        target_column=target_column,
        feature_columns=list(
            numeric_feature_data
        ),
        leakage_stages=leakage_stages,
        leakage_columns=leakage_columns,
        excluded_columns=excluded_columns,
    )