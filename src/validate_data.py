"""
Validate all raw MOE datasets before feature engineering.

Launched by:
    notebooks/01_feature_builder.ipynb

Purpose:
    - Validate required columns
    - Validate dataset grain
    - Validate date fields
    - Check duplicate rows
    - Check duplicate keys
    - Produce a validation report
    - Stop the pipeline if validation fails

Inputs:
    data/raw/*.csv

Output:
    data/outputs/validation_report.csv
"""

#=========================================================
# imports
#=========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.datasets import DATASETS
from config.settings import (
    OUTPUT_DIR,
    RAW_DIR,
)

#=========================================================
# paths
#=========================================================

validation_report_path = (
    OUTPUT_DIR
    / "validation_report.csv"
)


#=========================================================
# dataset contracts
#=========================================================

dataset_rules: dict[str, dict[str, Any]] = {
    DATASETS["dates"]: {
        "required_columns": [
            "date_day",
        ],
        "grain": [
            "date_day",
        ],
    },
    DATASETS["media"]: {
        "required_columns": [
            "date_day",
        ],
        "grain": [
            "date_day",
        ],
    },
    DATASETS["digital"]: {
        "required_columns": [
            "date_day",
            "platform",
            "channel",
            "metric",
            "value",
        ],
        "grain": [
            "date_day",
            "platform",
            "channel",
            "metric",
        ],
    },
    DATASETS["attribution"]: {
        "required_columns": [
            "date_day",
            "platform",
            "channel",
            "metric",
            "value",
        ],
        "grain": [
            "date_day",
            "platform",
            "channel",
            "metric",
        ],
    },
    DATASETS["external"]: {
        "required_columns": [
            "date_day",
        ],
        "grain": [
            "date_day",
        ],
    },
    DATASETS["funnel_uncohorted"]: {
        "required_columns": [
            "date_day",
        ],
        "grain": [
            "date_day",
        ],
    },
    DATASETS["funnel_cohorted"]: {
        "required_columns": [
            "date_day",
        ],
        "grain": [
            "date_day",
        ],
    },
}


#=========================================================
# helpers
#=========================================================

def _add_result(
    results: list[dict[str, Any]],
    dataset: str,
    check: str,
    status: str,
    details: str,
    affected_rows: int = 0,
) -> None:
    """Append one validation result."""

    results.append(
        {
            "dataset": dataset,
            "check": check,
            "status": status,
            "affected_rows": affected_rows,
            "details": details,
        }
    )


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with tidy lowercase column names."""

    output = df.copy()

    output.columns = (
        output.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return output


def _validate_file_exists(
    file_path: Path,
    results: list[dict[str, Any]],
) -> bool:
    """Validate that a required raw file exists."""

    if file_path.exists():
        _add_result(
            results=results,
            dataset=file_path.name,
            check="file_exists",
            status="pass",
            details="file found",
        )

        return True

    _add_result(
        results=results,
        dataset=file_path.name,
        check="file_exists",
        status="fail",
        details=f"file not found: {file_path}",
    )

    return False


def _validate_required_columns(
    df: pd.DataFrame,
    dataset: str,
    required_columns: list[str],
    results: list[dict[str, Any]],
) -> bool:
    """Validate that all required columns are present."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if not missing_columns:
        _add_result(
            results=results,
            dataset=dataset,
            check="required_columns",
            status="pass",
            details="all required columns found",
        )

        return True

    _add_result(
        results=results,
        dataset=dataset,
        check="required_columns",
        status="fail",
        details=f"missing columns: {', '.join(missing_columns)}",
    )

    return False


def _validate_dates(
    df: pd.DataFrame,
    dataset: str,
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Parse date_day and report invalid or future dates."""

    output = df.copy()

    parsed_dates = pd.to_datetime(
        output["date_day"],
        errors="coerce",
    )

    invalid_date_count = int(parsed_dates.isna().sum())

    if invalid_date_count == 0:
        _add_result(
            results=results,
            dataset=dataset,
            check="valid_dates",
            status="pass",
            details="all date_day values parsed successfully",
        )
    else:
        _add_result(
            results=results,
            dataset=dataset,
            check="valid_dates",
            status="fail",
            affected_rows=invalid_date_count,
            details="date_day contains invalid or missing values",
        )

    today = pd.Timestamp.now().normalize()
    future_date_count = int((parsed_dates > today).sum())

    if future_date_count == 0:
        _add_result(
            results=results,
            dataset=dataset,
            check="future_dates",
            status="pass",
            details="no future dates found",
        )
    else:
        _add_result(
            results=results,
            dataset=dataset,
            check="future_dates",
            status="fail",
            affected_rows=future_date_count,
            details="date_day contains future dates",
        )

    output["date_day"] = parsed_dates

    return output


def _validate_duplicate_rows(
    df: pd.DataFrame,
    dataset: str,
    results: list[dict[str, Any]],
) -> None:
    """Validate that complete duplicate rows are not present."""

    duplicate_row_count = int(df.duplicated().sum())

    if duplicate_row_count == 0:
        _add_result(
            results=results,
            dataset=dataset,
            check="duplicate_rows",
            status="pass",
            details="no complete duplicate rows found",
        )
    else:
        _add_result(
            results=results,
            dataset=dataset,
            check="duplicate_rows",
            status="fail",
            affected_rows=duplicate_row_count,
            details="complete duplicate rows found",
        )


def _validate_grain(
    df: pd.DataFrame,
    dataset: str,
    grain: list[str],
    results: list[dict[str, Any]],
) -> None:
    """Validate uniqueness at the agreed dataset grain."""

    missing_grain_columns = [
        column
        for column in grain
        if column not in df.columns
    ]

    if missing_grain_columns:
        _add_result(
            results=results,
            dataset=dataset,
            check="grain",
            status="fail",
            details=(
                "grain could not be checked because columns are missing: "
                + ", ".join(missing_grain_columns)
            ),
        )

        return

    duplicate_key_count = int(
        df.duplicated(
            subset=grain,
            keep=False,
        ).sum()
    )

    if duplicate_key_count == 0:
        _add_result(
            results=results,
            dataset=dataset,
            check="grain",
            status="pass",
            details=f"unique at grain: {', '.join(grain)}",
        )
    else:
        _add_result(
            results=results,
            dataset=dataset,
            check="grain",
            status="fail",
            affected_rows=duplicate_key_count,
            details=f"duplicate keys found at grain: {', '.join(grain)}",
        )


def _validate_value_column(
    df: pd.DataFrame,
    dataset: str,
    results: list[dict[str, Any]],
) -> None:
    """Validate the value column for long-format metric datasets."""

    if "value" not in df.columns:
        return

    clean_values = (
        df["value"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    
    numeric_values = pd.to_numeric(
        clean_values,
        errors="coerce",
    )

    invalid_value_count = int(
        numeric_values.isna().sum()
        - df["value"].isna().sum()
    )

    if invalid_value_count == 0:
        _add_result(
            results=results,
            dataset=dataset,
            check="numeric_value",
            status="pass",
            details="all non-null value entries are numeric",
        )
    else:
        _add_result(
            results=results,
            dataset=dataset,
            check="numeric_value",
            status="fail",
            affected_rows=invalid_value_count,
            details="value contains non-numeric entries",
        )


#=========================================================
# validation pipeline
#=========================================================

def validate_dataset(
    file_path: Path,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate one raw dataset against its data contract."""

    results: list[dict[str, Any]] = []

    if not _validate_file_exists(
        file_path=file_path,
        results=results,
    ):
        return results

    try:
        df = pd.read_csv(file_path)
        df = _normalise_column_names(df)

        _add_result(
            results=results,
            dataset=file_path.name,
            check="read_csv",
            status="pass",
            details=f"{len(df):,} rows loaded",
        )

    except Exception as exc:
        _add_result(
            results=results,
            dataset=file_path.name,
            check="read_csv",
            status="fail",
            details=f"{type(exc).__name__}: {exc}",
        )

        return results

    required_columns_present = _validate_required_columns(
        df=df,
        dataset=file_path.name,
        required_columns=rules["required_columns"],
        results=results,
    )

    _validate_duplicate_rows(
        df=df,
        dataset=file_path.name,
        results=results,
    )

    if required_columns_present and "date_day" in df.columns:
        df = _validate_dates(
            df=df,
            dataset=file_path.name,
            results=results,
        )

    _validate_grain(
        df=df,
        dataset=file_path.name,
        grain=rules["grain"],
        results=results,
    )

    _validate_value_column(
        df=df,
        dataset=file_path.name,
        results=results,
    )

    return results


def validate_raw_folder(
    raw_folder: Path | str = RAW_DIR,
    report_path: Path | str = validation_report_path,
) -> pd.DataFrame:
    """Validate every configured raw dataset and save one report."""

    raw_folder = Path(raw_folder)
    report_path = Path(report_path)

    all_results: list[dict[str, Any]] = []

    for dataset, rules in dataset_rules.items():
        file_path = raw_folder / dataset

        dataset_results = validate_dataset(
            file_path=file_path,
            rules=rules,
        )

        all_results.extend(dataset_results)

    report = pd.DataFrame(all_results)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        report_path,
        index=False,
    )

    return report


def assert_validation_passed(
    report: pd.DataFrame | None = None,
    report_path: Path | str = validation_report_path,
) -> None:
    """Raise an error when any validation check has failed."""

    if report is None:
        report_path = Path(report_path)

        if not report_path.exists():
            raise FileNotFoundError(
                f"validation report not found: {report_path}"
            )

        report = pd.read_csv(report_path)

    failed_checks = report.loc[
        report["status"].eq("fail")
    ]

    if not failed_checks.empty:
        failed_summary = failed_checks[
            [
                "dataset",
                "check",
                "details",
            ]
        ].to_string(index=False)

        raise ValueError(
            "raw data validation failed:\n\n"
            + failed_summary
        )


if __name__ == "__main__":
    validation_report = validate_raw_folder()

    print("\n=====================================")
    print("==validation report")
    print(validation_report)

    assert_validation_passed(validation_report)

    print("\nall validation checks passed")
