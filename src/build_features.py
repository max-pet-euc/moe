"""
Build the canonical daily MOE feature dataset.

Launched by:
    notebooks/01_feature_builder.ipynb

Purpose:
    - Load validated raw datasets
    - Pivot long-format digital platform metrics
    - Pivot long-format attribution metrics
    - Merge every dataset onto the canonical date table
    - Save one daily feature dataset

Inputs:
    data/raw/data_dates.csv
    data/raw/data_media_inputs.csv
    data/raw/data_digital_platforms.csv
    data/raw/data_attribution.csv
    data/raw/data_external.csv
    data/raw/data_funnel_uncohorted.csv
    data/raw/data_funnel_cohorted.csv

Output:
    data/engineered/data_features.csv
"""

#=========================================================
# imports
#=========================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.datasets import DATASETS
from config.settings import (
    ENGINEERED_DIR,
    RAW_DIR,
)

#=========================================================
# paths
#=========================================================

feature_output_path = (
    ENGINEERED_DIR
    / "data_features.csv"
)


#=========================================================
# dataset configuration
#=========================================================

wide_datasets = {
    "dates": DATASETS["dates"],
    "media": DATASETS["media"],
    "external": DATASETS["external"],
    "funnel_uncohorted": DATASETS["funnel_uncohorted"],
    "funnel_cohorted": DATASETS["funnel_cohorted"],
}

long_datasets = {
    "digital": DATASETS["digital"],
    "attribution": DATASETS["attribution"],
}


#=========================================================
# helpers
#=========================================================

def _clean_name(value: object) -> str:
    """Convert a value into a safe lowercase feature-name component."""

    cleaned = str(value).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned.strip("_")


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with tidy lowercase column names."""

    output = df.copy()

    output.columns = [
        _clean_name(column)
        for column in output.columns
    ]

    return output


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    """Load one CSV and standardise its schema."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"required input file not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    df = _normalise_column_names(df)

    if "date_day" not in df.columns:
        raise ValueError(
            f"{file_path.name} does not contain date_day"
        )

    df["date_day"] = pd.to_datetime(
        df["date_day"],
        errors="raise",
    )

    return df


def prefix_wide_columns(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Prefix all non-key columns in a wide daily dataset."""

    rename_map = {
        column: f"{prefix}_{column}"
        for column in df.columns
        if column != "date_day"
    }

    return df.rename(columns=rename_map)


def pivot_metric_table(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Pivot a date × platform × channel × metric table into daily columns.

    Example output:
        digital_google_youtube_impressions
        attribution_google_search_paid_last_click_qs
    """

    required_columns = {
        "date_day",
        "platform",
        "channel",
        "metric",
        "value",
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "long metric table is missing columns: "
            + ", ".join(missing_columns)
        )

    output = df.copy()

    output["platform"] = output["platform"].map(_clean_name)
    output["channel"] = output["channel"].map(_clean_name)
    output["metric"] = output["metric"].map(_clean_name)

    output["feature_name"] = (
        prefix
        + "_"
        + output["platform"]
        + "_"
        + output["channel"]
        + "_"
        + output["metric"]
    )

    output["value"] = pd.to_numeric(
        output["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="raise",
    )

    duplicate_grain = [
        "date_day",
        "feature_name",
    ]

    duplicate_rows = output.duplicated(
        subset=duplicate_grain,
        keep=False,
    )

    if duplicate_rows.any():
        duplicate_examples = (
            output.loc[
                duplicate_rows,
                duplicate_grain,
            ]
            .drop_duplicates()
            .head(10)
            .to_string(index=False)
        )

        raise ValueError(
            "duplicate long-metric keys found after feature naming:\n\n"
            + duplicate_examples
        )

    pivoted = (
        output
        .pivot(
            index="date_day",
            columns="feature_name",
            values="value",
        )
        .reset_index()
    )

    pivoted.columns.name = None

    return pivoted


def merge_daily_dataset(
    feature_df: pd.DataFrame,
    new_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Left join one daily dataset onto the canonical feature table."""

    if not new_df["date_day"].is_unique:
        raise ValueError(
            f"{dataset_name} is not unique by date_day"
        )

    overlapping_columns = sorted(
        set(feature_df.columns)
        .intersection(new_df.columns)
        .difference({"date_day"})
    )

    if overlapping_columns:
        raise ValueError(
            f"{dataset_name} contains overlapping columns: "
            + ", ".join(overlapping_columns)
        )

    return feature_df.merge(
        new_df,
        on="date_day",
        how="left",
        validate="one_to_one",
    )


def validate_feature_output(
    feature_df: pd.DataFrame,
) -> None:
    """Run core checks before saving the canonical feature dataset."""

    if feature_df.empty:
        raise ValueError(
            "feature dataset is empty"
        )

    if feature_df["date_day"].isna().any():
        raise ValueError(
            "feature dataset contains missing date_day values"
        )

    if not feature_df["date_day"].is_unique:
        raise ValueError(
            "feature dataset contains duplicate date_day values"
        )

    numeric_df = feature_df.select_dtypes(
        include="number"
    )

    infinity_count = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    if infinity_count > 0:
        raise ValueError(
            f"feature dataset contains {infinity_count:,} infinite values"
        )

    duplicate_columns = feature_df.columns[
        feature_df.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            "feature dataset contains duplicate columns: "
            + ", ".join(duplicate_columns)
        )


#=========================================================
# feature build
#=========================================================

def build_features(
    input_folder: Path | str = raw_dir,
    output_path: Path | str = feature_output_path,
) -> pd.DataFrame:
    """Build and save the canonical daily MOE feature dataset."""

    input_folder = Path(input_folder)
    output_path = Path(output_path)

    #=====================================================
    # load canonical date table
    #=====================================================

    feature_df = load_csv(
        input_folder / wide_datasets["dates"]
    )[["date_day"]].copy()

    if not feature_df["date_day"].is_unique:
        raise ValueError(
            "data_dates.csv must contain one row per date_day"
        )

    feature_df = feature_df.sort_values(
        "date_day"
    ).reset_index(
        drop=True
    )

    #=====================================================
    # merge wide daily datasets
    #=====================================================

    wide_prefixes = {
        "media_inputs": "",
        "external": "external",
        "funnel_uncohorted": "uncohorted",
        "funnel_cohorted": "cohorted",
    }

    for dataset_name, prefix in wide_prefixes.items():
        filename = wide_datasets[dataset_name]

        dataset_df = load_csv(
            input_folder / filename
        )

        if prefix:
            dataset_df = prefix_wide_columns(
                df=dataset_df,
                prefix=prefix,
            )

        feature_df = merge_daily_dataset(
            feature_df=feature_df,
            new_df=dataset_df,
            dataset_name=dataset_name,
        )

    #=====================================================
    # pivot and merge long metric datasets
    #=====================================================

    for prefix, filename in long_datasets.items():
        dataset_df = load_csv(
            input_folder / filename
        )

        dataset_df = pivot_metric_table(
            df=dataset_df,
            prefix=prefix,
        )

        feature_df = merge_daily_dataset(
            feature_df=feature_df,
            new_df=dataset_df,
            dataset_name=prefix,
        )

    #=====================================================
    # validate and save
    #=====================================================

    feature_df = feature_df.sort_values(
        "date_day"
    ).reset_index(
        drop=True
    )

    validate_feature_output(feature_df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        output_path,
        index=False,
    )

    return feature_df


if __name__ == "__main__":
    features = build_features()

    print("\n=====================================")
    print("==feature dataset")
    print(f"rows: {len(features):,}")
    print(f"columns: {len(features.columns):,}")
    print(f"date min: {features['date_day'].min().date()}")
    print(f"date max: {features['date_day'].max().date()}")
    print(f"saved to: {feature_output_path}")
