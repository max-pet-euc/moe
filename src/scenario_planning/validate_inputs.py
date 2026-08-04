# src/scenario_planning/validate_inputs.py
# PLANNING_INPUT_VALIDATION_VERSION = "1.0-20260803_2222"

"""
Validate monthly scenario-planning inputs and build Scenario objects.

Expected files:
- data/planning_inputs/budgets.csv
- data/planning_inputs/targets.csv

budgets.csv:
    date_month plus one or more budget_* channel columns

targets.csv:
    date_month, uncohorted_qs, uncohorted_op, budget_total
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .scenario import (
    CommercialTargets,
    GrowthBudgetPlan,
    Scenario,
    ScenarioStatus,
    ScenarioType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLANNING_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "planning_inputs"
)

BUDGETS_PATH = (
    PLANNING_INPUT_DIR
    / "budgets.csv"
)

TARGETS_PATH = (
    PLANNING_INPUT_DIR
    / "targets.csv"
)

BUDGET_TOLERANCE = 1.00

REQUIRED_TARGET_COLUMNS = {
    "date_month",
    "target_uncohorted_qs",
    "target_uncohorted_op",
    "target_budget_media",
}

NON_MEDIA_BUDGET_COLUMNS = {
    "budget_creator",
}


class PlanningInputValidationError(
    ValueError
):
    """Raised when planning inputs cannot safely build scenarios."""


@dataclass(frozen=True)
class PlanningValidationResult:
    """Validated source tables and the scenarios built from them."""

    budgets: pd.DataFrame
    targets: pd.DataFrame
    scenarios: list[Scenario]
    warnings: tuple[str, ...]


def clean_numeric_series(
    series: pd.Series,
) -> pd.Series:
    """Convert comma-formatted planning values to numeric."""

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
            }
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def load_csv_as_strings(
    path: Path,
) -> pd.DataFrame:
    """Load a planning CSV without inferring formatted numbers."""

    if not path.exists():
        raise PlanningInputValidationError(
            f"planning input file not found: {path}"
        )

    try:
        return pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
        )
    except Exception as exc:
        raise PlanningInputValidationError(
            f"could not read planning input file "
            f"{path.name}: {exc}"
        ) from exc


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Raise when required columns are missing."""

    missing_columns = sorted(
        set(required_columns)
        .difference(df.columns)
    )

    if missing_columns:
        raise PlanningInputValidationError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def parse_month_column(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Parse date_month and require first-of-month values."""

    output = df.copy()

    output["date_month"] = pd.to_datetime(
        output["date_month"],
        errors="coerce",
    )

    invalid_dates = output[
        "date_month"
    ].isna()

    if invalid_dates.any():
        row_numbers = (
            output.index[
                invalid_dates
            ]
            .add(2)
            .tolist()
        )

        raise PlanningInputValidationError(
            f"{dataset_name} has invalid date_month "
            f"values on CSV rows: {row_numbers}"
        )

    non_month_start = (
        output["date_month"].dt.day
        != 1
    )

    if non_month_start.any():
        bad_dates = (
            output.loc[
                non_month_start,
                "date_month",
            ]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise PlanningInputValidationError(
            f"{dataset_name} date_month must be "
            f"the first day of each month: {bad_dates}"
        )

    duplicates = output[
        "date_month"
    ].duplicated(
        keep=False
    )

    if duplicates.any():
        duplicate_months = (
            output.loc[
                duplicates,
                "date_month",
            ]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )

        raise PlanningInputValidationError(
            f"{dataset_name} contains duplicate months: "
            f"{duplicate_months}"
        )

    return (
        output
        .sort_values("date_month")
        .reset_index(drop=True)
    )


def validate_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    dataset_name: str,
    *,
    allow_null: bool = False,
) -> pd.DataFrame:
    """Clean and validate configured numeric columns."""

    output = df.copy()

    for column in numeric_columns:
        raw_values = output[column].copy()

        output[column] = (
            clean_numeric_series(
                output[column]
            )
        )

        invalid_numeric = (
            raw_values.astype("string")
            .str.strip()
            .ne("")
            & output[column].isna()
        )

        if invalid_numeric.any():
            bad_values = (
                raw_values.loc[
                    invalid_numeric
                ]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            raise PlanningInputValidationError(
                f"{dataset_name}.{column} contains "
                f"non-numeric values: {bad_values}"
            )

        if (
            not allow_null
            and output[column].isna().any()
        ):
            missing_months = (
                output.loc[
                    output[column].isna(),
                    "date_month",
                ]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            raise PlanningInputValidationError(
                f"{dataset_name}.{column} is missing "
                f"for months: {missing_months}"
            )

        negative_values = (
            output[column].notna()
            & output[column].lt(0)
        )

        if negative_values.any():
            negative_months = (
                output.loc[
                    negative_values,
                    "date_month",
                ]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            raise PlanningInputValidationError(
                f"{dataset_name}.{column} contains "
                f"negative values for months: "
                f"{negative_months}"
            )

    return output


def validate_budgets(
    budgets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate Growth's monthly channel budget plan."""

    validate_required_columns(
        budgets,
        {"date_month"},
        "budgets.csv",
    )

    output = parse_month_column(
        budgets,
        "budgets.csv",
    )

    budget_columns = [
        column
        for column in output.columns
        if column.startswith("budget_")
    ]

    if not budget_columns:
        raise PlanningInputValidationError(
            "budgets.csv must contain at least "
            "one budget_* channel column"
        )

    required_summary_columns = {
        "budget_total",
        "budget_media",
    }

    missing_summary_columns = sorted(
        required_summary_columns.difference(
            output.columns
        )
    )

    if missing_summary_columns:
        raise PlanningInputValidationError(
            "budgets.csv is missing columns: "
            + ", ".join(
                missing_summary_columns
            )
        )

    invalid_budget_names = [
        column
        for column in budget_columns
        if not re.fullmatch(
            r"budget_[a-z0-9_]+",
            column,
        )
    ]

    if invalid_budget_names:
        raise PlanningInputValidationError(
            "budgets.csv contains invalid budget "
            "column names: "
            + ", ".join(invalid_budget_names)
        )

    unexpected_columns = sorted(
        set(output.columns)
        .difference(
            {"date_month", *budget_columns}
        )
    )

    if unexpected_columns:
        raise PlanningInputValidationError(
            "budgets.csv contains unexpected columns: "
            + ", ".join(unexpected_columns)
        )

    output = validate_numeric_columns(
        output,
        budget_columns,
        "budgets.csv",
    )
    
    detail_budget_columns = [
        column
        for column in budget_columns
        if column not in {
            "budget_total",
            "budget_media",
        }
    ]

    media_budget_columns = [
        column
        for column in detail_budget_columns
        if column
        not in NON_MEDIA_BUDGET_COLUMNS
    ]

    output[
        "allocated_budget_total"
    ] = output[
        detail_budget_columns
    ].sum(axis=1)

    output[
        "allocated_budget_media"
    ] = output[
        media_budget_columns
    ].sum(axis=1)

    warnings: list[str] = []

    zero_budget_columns = [
        column
        for column in budget_columns
        if output[column].fillna(0).eq(0).all()
    ]

    if zero_budget_columns:
        warnings.append(
            "These budget columns are zero for every "
            "month: "
            + ", ".join(zero_budget_columns)
        )

    return output, warnings


def validate_targets(
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate Commercial targets and approved budgets."""

    validate_required_columns(
        targets,
        REQUIRED_TARGET_COLUMNS,
        "targets.csv",
    )

    unexpected_columns = sorted(
        set(targets.columns)
        .difference(
            REQUIRED_TARGET_COLUMNS
        )
    )

    if unexpected_columns:
        raise PlanningInputValidationError(
            "targets.csv contains unexpected columns: "
            + ", ".join(unexpected_columns)
        )

    output = parse_month_column(
        targets,
        "targets.csv",
    )

    output = validate_numeric_columns(
        output,
        [
            "target_uncohorted_qs",
            "target_uncohorted_op",
            "target_budget_media",
        ],
        "targets.csv",
    )

    warnings: list[str] = []

    for column in [
        "target_uncohorted_qs",
        "target_uncohorted_op",
    ]:
        zero_targets = output[
            column
        ].eq(0)

        if zero_targets.any():
            months = (
                output.loc[
                    zero_targets,
                    "date_month",
                ]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            warnings.append(
                f"{column} is zero for months: "
                f"{months}"
            )

    return output, warnings


def validate_month_alignment(
    budgets: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """Require exactly the same months in both planning files."""

    budget_months = set(
        budgets["date_month"]
    )

    target_months = set(
        targets["date_month"]
    )

    missing_from_budgets = sorted(
        target_months
        - budget_months
    )

    missing_from_targets = sorted(
        budget_months
        - target_months
    )

    errors: list[str] = []

    if missing_from_budgets:
        errors.append(
            "months in targets.csv but not budgets.csv: "
            + ", ".join(
                month.strftime("%Y-%m-%d")
                for month in missing_from_budgets
            )
        )

    if missing_from_targets:
        errors.append(
            "months in budgets.csv but not targets.csv: "
            + ", ".join(
                month.strftime("%Y-%m-%d")
                for month in missing_from_targets
            )
        )

    if errors:
        raise PlanningInputValidationError(
            "; ".join(errors)
        )


def validate_budget_reconciliation(
    budgets: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate Growth totals and Commercial media alignment.
    """

    comparison = (
        targets[
            [
                "date_month",
                "target_budget_media",
            ]
        ]
        .merge(
            budgets[
                [
                    "date_month",
                    "budget_total",
                    "budget_media",
                    "allocated_budget_total",
                    "allocated_budget_media",
                ]
            ],
            on="date_month",
            how="inner",
            validate="one_to_one",
        )
    )

    comparison[
        "budget_total_variance"
    ] = (
        comparison[
            "allocated_budget_total"
        ]
        - comparison[
            "budget_total"
        ]
    )

    comparison[
        "budget_media_variance"
    ] = (
        comparison[
            "allocated_budget_media"
        ]
        - comparison[
            "budget_media"
        ]
    )

    comparison[
        "media_target_variance"
    ] = (
        comparison[
            "budget_media"
        ]
        - comparison[
            "target_budget_media"
        ]
    )

    validation_columns = [
        "budget_total_variance",
        "budget_media_variance",
        "media_target_variance",
    ]

    failed = comparison[
        validation_columns
    ].abs().gt(
        BUDGET_TOLERANCE
    ).any(axis=1)

    if failed.any():
        details = (
            comparison.loc[
                failed,
                [
                    "date_month",
                    "target_budget_media",
                    "budget_media",
                    "allocated_budget_media",
                    "budget_total",
                    "allocated_budget_total",
                    "media_target_variance",
                    "budget_media_variance",
                    "budget_total_variance",
                ],
            ]
            .assign(
                date_month=lambda frame: (
                    frame[
                        "date_month"
                    ]
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                )
            )
            .to_string(index=False)
        )

        raise PlanningInputValidationError(
            "Planning budgets do not reconcile "
            f"within £{BUDGET_TOLERANCE:.2f}:\n\n"
            + details
        )

    return comparison


def make_scenario_id(
    date_month: pd.Timestamp,
    scenario_type: ScenarioType,
) -> str:
    """Create a stable scenario identifier."""

    return (
        f"{date_month:%Y_%m}_"
        f"{scenario_type.value}"
    )


def build_scenarios(
    budgets: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    scenario_type: ScenarioType = (
        ScenarioType.MANUAL
    ),
) -> list[Scenario]:
    """Build one validated Scenario object per month."""

    budget_columns = [
        column
        for column in budgets.columns
        if (
            column.startswith("budget_")
            and column not in {
                "budget_total",
                "budget_media",
            }
        )
    ]

    merged = (
        targets
        .merge(
            budgets.drop(
                columns=[
                    "allocated_budget_total"
                ],
                errors="ignore",
            ),
            on="date_month",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("date_month")
        .reset_index(drop=True)
    )

    scenarios: list[Scenario] = []

    for row in merged.itertuples(
        index=False
    ):
        row_values = row._asdict()

        date_month = pd.Timestamp(
            row_values["date_month"]
        )

        commercial_targets = CommercialTargets(
            date_month=date_month,
            target_uncohorted_qs=float(
                row_values[
                    "target_uncohorted_qs"
                ]
            ),
            target_uncohorted_op=float(
                row_values[
                    "target_uncohorted_op"
                ]
            ),
            target_budget_media=float(
                row_values[
                    "target_budget_media"
                ]
            ),
        )

        channel_budgets = {
            column: float(
                row_values[column]
            )
            for column in budget_columns
        }

        growth_budget_plan = GrowthBudgetPlan(
            date_month=date_month,
            budget_total=float(
                row_values["budget_total"]
            ),
            budget_media=float(
                row_values["budget_media"]
            ),
            channel_budgets=channel_budgets,
        )

        scenario = Scenario(
            scenario_id=make_scenario_id(
                date_month,
                scenario_type,
            ),
            scenario_name=(
                f"{date_month:%B %Y} "
                f"{scenario_type.value}"
            ),
            commercial_targets=(
                commercial_targets
            ),
            growth_budget_plan=(
                growth_budget_plan
            ),
            scenario_type=scenario_type,
            status=(
                ScenarioStatus.VALIDATED
            ),
        )

        scenarios.append(
            scenario
        )

    return scenarios


def load_and_validate_planning_inputs(
    budgets_path: Path = BUDGETS_PATH,
    targets_path: Path = TARGETS_PATH,
) -> PlanningValidationResult:
    """Load, validate and convert planning files into scenarios."""

    raw_budgets = load_csv_as_strings(
        budgets_path
    )

    raw_targets = load_csv_as_strings(
        targets_path
    )

    budgets, budget_warnings = (
        validate_budgets(
            raw_budgets
        )
    )

    targets, target_warnings = (
        validate_targets(
            raw_targets
        )
    )

    validate_month_alignment(
        budgets,
        targets,
    )

    validate_budget_reconciliation(
        budgets,
        targets,
    )

    scenarios = build_scenarios(
        budgets,
        targets,
    )

    return PlanningValidationResult(
        budgets=budgets,
        targets=targets,
        scenarios=scenarios,
        warnings=tuple(
            [
                *budget_warnings,
                *target_warnings,
            ]
        ),
    )


def print_validation_summary(
    result: PlanningValidationResult,
) -> None:
    """Print a compact planning-input validation report."""

    scenario_summary = pd.DataFrame(
        scenario.to_summary_dict()
        for scenario in result.scenarios
    )

    display_columns = [
        "scenario_id",
        "date_month",
        "target_uncohorted_qs",
        "target_uncohorted_op",
        "target_budget_media",
        "budget_media",
        "media_target_variance",
        "budget_total",
        "allocated_budget_total",
        "budget_total_variance",
    ]

    print(
        "\n====================================="
    )
    print(
        "==planning input validation"
    )
    print(
        f"months validated: "
        f"{len(result.scenarios)}"
    )
    print(
        f"first month: "
        f"{scenario_summary['date_month'].min()}"
    )
    print(
        f"last month: "
        f"{scenario_summary['date_month'].max()}"
    )

    print(
        "\n==scenario objects"
    )
    print(
        scenario_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    if result.warnings:
        print(
            "\n==warnings"
        )

        for warning in result.warnings:
            print(
                f"- {warning}"
            )

    print(
        "\nall planning input checks passed"
    )


def main() -> None:
    """Command-line entry point."""

    result = (
        load_and_validate_planning_inputs()
    )

    print_validation_summary(
        result
    )


if __name__ == "__main__":
    try:
        main()
    except PlanningInputValidationError as exc:
        raise SystemExit(
            f"planning input validation failed:\n\n"
            f"{exc}"
        ) from exc
