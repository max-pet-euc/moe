"""
Run quality checks on the canonical daily MOE feature dataset.

Launched by:
    notebooks/01_feature_builder.ipynb

Purpose:
    - Confirm one row per date
    - Check date coverage and missing dates
    - Check duplicate columns
    - Check null rates
    - Check infinite values
    - Summarise numeric feature distributions
    - Save feature quality outputs

Input:
    data/engineered/data_features.csv

Outputs:
    data/outputs/feature_quality_report.csv
    data/outputs/feature_summary.csv
"""

#=========================================================
# imports
#=========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


#=========================================================
# paths
#=========================================================

project_root = Path(__file__).resolve().parents[1]

feature_input_path = (
    project_root
    / "data"
    / "engineered"
    / "data_features.csv"
)

output_dir = project_root / "data" / "outputs"

quality_report_path = (
    output_dir
    / "feature_quality_report.csv"
)

feature_summary_path = (
    output_dir
    / "feature_summary.csv"
)


#=========================================================
# helpers
#=========================================================

def _add_result(
    results: list[dict[str, Any]],
    check: str,
    status: str,
    details: str,
    affected_rows: int = 0,
) -> None:
    """Append one quality-check result."""

    results.append(
        {
            "check": check,
            "status": status,
            "affected_rows": affected_rows,
            "details": details,
        }
    )


def load_features(
    file_path: Path | str = feature_input_path,
) -> pd.DataFrame:
    """Load the engineered feature dataset."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"feature dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if "date_day" not in df.columns:
        raise ValueError(
            "feature dataset does not contain date_day"
        )

    df["date_day"] = pd.to_datetime(
        df["date_day"],
        errors="raise",
    )

    return df


#=========================================================
# feature quality checks
#=========================================================

def run_feature_quality_checks(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run structural and value-quality checks."""

    results: list[dict[str, Any]] = []

    #=====================================================
    # row count
    #=====================================================

    if feature_df.empty:
        _add_result(
            results=results,
            check="row_count",
            status="fail",
            details="feature dataset contains no rows",
        )
    else:
        _add_result(
            results=results,
            check="row_count",
            status="pass",
            details=f"{len(feature_df):,} rows found",
        )

    #=====================================================
    # date checks
    #=====================================================

    missing_date_count = int(
        feature_df["date_day"].isna().sum()
    )

    if missing_date_count == 0:
        _add_result(
            results=results,
            check="missing_date_day",
            status="pass",
            details="no missing date_day values",
        )
    else:
        _add_result(
            results=results,
            check="missing_date_day",
            status="fail",
            affected_rows=missing_date_count,
            details="missing date_day values found",
        )

    duplicate_date_count = int(
        feature_df.duplicated(
            subset=["date_day"],
            keep=False,
        ).sum()
    )

    if duplicate_date_count == 0:
        _add_result(
            results=results,
            check="duplicate_date_day",
            status="pass",
            details="one row per date_day",
        )
    else:
        _add_result(
            results=results,
            check="duplicate_date_day",
            status="fail",
            affected_rows=duplicate_date_count,
            details="duplicate date_day values found",
        )

    sorted_dates = (
        feature_df["date_day"]
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if sorted_dates.empty:
        missing_dates = pd.DatetimeIndex([])
    else:
        expected_dates = pd.date_range(
            start=sorted_dates.min(),
            end=sorted_dates.max(),
            freq="D",
        )

        missing_dates = expected_dates.difference(
            sorted_dates
        )

    if len(missing_dates) == 0:
        _add_result(
            results=results,
            check="date_continuity",
            status="pass",
            details="no missing dates within the feature range",
        )
    else:
        sample_dates = ", ".join(
            date.strftime("%Y-%m-%d")
            for date in missing_dates[:10]
        )

        _add_result(
            results=results,
            check="date_continuity",
            status="fail",
            affected_rows=len(missing_dates),
            details=(
                f"missing dates found; first examples: {sample_dates}"
            ),
        )

    #=====================================================
    # column checks
    #=====================================================

    duplicate_columns = (
        feature_df.columns[
            feature_df.columns.duplicated()
        ]
        .tolist()
    )

    if not duplicate_columns:
        _add_result(
            results=results,
            check="duplicate_columns",
            status="pass",
            details="no duplicate column names",
        )
    else:
        _add_result(
            results=results,
            check="duplicate_columns",
            status="fail",
            affected_rows=len(duplicate_columns),
            details=(
                "duplicate columns found: "
                + ", ".join(duplicate_columns)
            ),
        )

    unnamed_columns = [
        column
        for column in feature_df.columns
        if str(column).lower().startswith("unnamed")
    ]

    if not unnamed_columns:
        _add_result(
            results=results,
            check="unnamed_columns",
            status="pass",
            details="no unnamed index columns",
        )
    else:
        _add_result(
            results=results,
            check="unnamed_columns",
            status="fail",
            affected_rows=len(unnamed_columns),
            details=(
                "unexpected unnamed columns found: "
                + ", ".join(unnamed_columns)
            ),
        )

    #=====================================================
    # numeric checks
    #=====================================================

    numeric_df = feature_df.select_dtypes(
        include="number"
    )

    infinite_count = int(
        np.isinf(
            numeric_df.to_numpy()
        ).sum()
    )

    if infinite_count == 0:
        _add_result(
            results=results,
            check="infinite_values",
            status="pass",
            details="no positive or negative infinite values",
        )
    else:
        _add_result(
            results=results,
            check="infinite_values",
            status="fail",
            affected_rows=infinite_count,
            details="infinite numeric values found",
        )

    entirely_null_columns = [
        column
        for column in feature_df.columns
        if feature_df[column].isna().all()
    ]

    if not entirely_null_columns:
        _add_result(
            results=results,
            check="entirely_null_columns",
            status="pass",
            details="no columns are entirely null",
        )
    else:
        _add_result(
            results=results,
            check="entirely_null_columns",
            status="fail",
            affected_rows=len(entirely_null_columns),
            details=(
                "entirely null columns found: "
                + ", ".join(entirely_null_columns)
            ),
        )

    constant_numeric_columns = [
        column
        for column in numeric_df.columns
        if numeric_df[column].nunique(
            dropna=True
        ) <= 1
    ]

    if not constant_numeric_columns:
        _add_result(
            results=results,
            check="constant_numeric_columns",
            status="pass",
            details="no constant numeric features",
        )
    else:
        _add_result(
            results=results,
            check="constant_numeric_columns",
            status="warning",
            affected_rows=len(constant_numeric_columns),
            details=(
                "constant numeric features found: "
                + ", ".join(constant_numeric_columns)
            ),
        )

    return pd.DataFrame(results)


#=========================================================
# feature summary
#=========================================================

def build_feature_summary(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row per feature column."""

    summary_rows: list[dict[str, Any]] = []

    total_rows = len(feature_df)

    for column in feature_df.columns:
        series = feature_df[column]

        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": total_rows,
            "non_null_rows": int(series.notna().sum()),
            "null_rows": int(series.isna().sum()),
            "null_percentage": round(
                float(series.isna().mean() * 100),
                2,
            ),
            "unique_values": int(
                series.nunique(
                    dropna=True
                )
            ),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "zero_rows": None,
            "negative_rows": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            row["minimum"] = numeric_series.min()
            row["maximum"] = numeric_series.max()
            row["mean"] = numeric_series.mean()
            row["median"] = numeric_series.median()
            row["standard_deviation"] = numeric_series.std()
            row["zero_rows"] = int(
                numeric_series.eq(0).sum()
            )
            row["negative_rows"] = int(
                numeric_series.lt(0).sum()
            )

        elif pd.api.types.is_datetime64_any_dtype(series):
            row["minimum"] = series.min()
            row["maximum"] = series.max()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


#=========================================================
# quality pipeline
#=========================================================

def create_quality_outputs(
    feature_df: pd.DataFrame | None = None,
    feature_path: Path | str = feature_input_path,
    report_path: Path | str = quality_report_path,
    summary_path: Path | str = feature_summary_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run checks, save reports and return both outputs."""

    if feature_df is None:
        feature_df = load_features(
            file_path=feature_path
        )

    report = run_feature_quality_checks(
        feature_df=feature_df
    )

    summary = build_feature_summary(
        feature_df=feature_df
    )

    report_path = Path(report_path)
    summary_path = Path(summary_path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        report_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return report, summary


def assert_feature_quality_passed(
    report: pd.DataFrame,
) -> None:
    """Raise an error when a feature-quality check fails."""

    failed_checks = report.loc[
        report["status"].eq("fail")
    ]

    if failed_checks.empty:
        return

    failed_summary = failed_checks[
        [
            "check",
            "affected_rows",
            "details",
        ]
    ].to_string(
        index=False
    )

    raise ValueError(
        "feature quality checks failed:\n\n"
        + failed_summary
    )


if __name__ == "__main__":
    quality_report, feature_summary = (
        create_quality_outputs()
    )

    print("\n=====================================")
    print("==feature quality report")
    print(quality_report)

    assert_feature_quality_passed(
        quality_report
    )

    print("\n=====================================")
    print("==feature summary")
    print(feature_summary.head(20))

    print("\nall feature quality checks passed")
