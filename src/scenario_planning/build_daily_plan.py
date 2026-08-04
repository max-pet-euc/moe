# src/scenario_planning/build_daily_plan.py
# DAILY_SCENARIO_PLAN_VERSION = "1.0-20260804_1147"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .scenario import Scenario
from .validate_inputs import (
    PlanningInputValidationError,
    load_and_validate_planning_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "scenarios"
DAILY_PLAN_OUTPUT_PATH = SCENARIO_OUTPUT_DIR / "scenario_daily_plan.csv"
MONTHLY_SUMMARY_OUTPUT_PATH = SCENARIO_OUTPUT_DIR / "scenario_monthly_summary.csv"
CURRENCY_DECIMALS = 2
RECONCILIATION_TOLERANCE = 0.01


class DailyPlanBuildError(ValueError):
    """Raised when a daily plan cannot be built safely."""


@dataclass(frozen=True)
class DailyPlanBuildResult:
    daily_plan: pd.DataFrame
    monthly_summary: pd.DataFrame
    daily_output_path: Path
    monthly_output_path: Path

def get_flighting_weights(
    flighting_profiles: pd.DataFrame,
    *,
    profile_name: str,
    days_in_month: int,
) -> list[float]:
    """Return ordered daily weights for one profile."""

    profile = (
        flighting_profiles.loc[
            flighting_profiles[
                "profile_name"
            ].eq(profile_name)
            & flighting_profiles[
                "days_in_month"
            ].eq(days_in_month)
        ]
        .sort_values(
            "day_of_month"
        )
    )

    if profile.empty:
        raise DailyPlanBuildError(
            f"flighting profile not found: "
            f"profile={profile_name}, "
            f"days={days_in_month}"
        )

    return (
        profile["weight"]
        .astype(float)
        .tolist()
    )

def allocate_monthly_value_by_weights(
    monthly_value: float,
    weights: list[float],
    *,
    decimals: int = CURRENCY_DECIMALS,
) -> list[float]:
    """Allocate a monthly amount using daily weights."""

    value = round(
        float(monthly_value),
        decimals,
    )

    allocation = [
        round(
            value * weight,
            decimals,
        )
        for weight in weights
    ]

    remainder = round(
        value - sum(allocation),
        decimals,
    )

    allocation[-1] = round(
        allocation[-1] + remainder,
        decimals,
    )

    return allocation

def allocate_monthly_value_evenly(
    monthly_value: float,
    days: int,
    *,
    decimals: int = CURRENCY_DECIMALS,
) -> list[float]:
    """Allocate a monthly amount evenly and preserve its exact total."""

    if days <= 0:
        raise DailyPlanBuildError(f"days must be positive; received {days}")

    value = round(float(monthly_value), decimals)
    base_daily_value = round(value / days, decimals)
    allocation = [base_daily_value for _ in range(days)]
    remainder = round(value - round(sum(allocation), decimals), decimals)
    allocation[-1] = round(allocation[-1] + remainder, decimals)

    if abs(round(sum(allocation), decimals) - value) > RECONCILIATION_TOLERANCE:
        raise DailyPlanBuildError(
            f"daily allocation did not reconcile: monthly={value}, "
            f"daily_total={round(sum(allocation), decimals)}"
        )

    return allocation


def build_daily_scenario_plan(
    scenario: Scenario,
    flighting_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Expand one validated monthly Scenario into daily rows."""

    if scenario.status.value != "validated":
        raise DailyPlanBuildError(
            f"scenario must be validated before daily expansion: "
            f"{scenario.scenario_id} has status "
            f"{scenario.status.value}"
        )

    flighting_weights = get_flighting_weights(
        flighting_profiles,
        profile_name=scenario.flighting_profile,
        days_in_month=scenario.days,
    )

    dates = pd.date_range(
        start=scenario.date_month,
        end=(
            scenario.date_month
            + pd.offsets.MonthEnd(0)
        ),
        freq="D",
    )

    if len(dates) != scenario.days:
        raise DailyPlanBuildError(
            f"calendar day mismatch for "
            f"{scenario.scenario_id}: "
            f"scenario days={scenario.days}, "
            f"generated days={len(dates)}"
        )

    if len(flighting_weights) != scenario.days:
        raise DailyPlanBuildError(
            f"flighting weight count mismatch for "
            f"{scenario.scenario_id}: "
            f"expected {scenario.days}, "
            f"found {len(flighting_weights)}"
        )

    daily_plan = pd.DataFrame(
        {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.scenario_name,
            "scenario_type": scenario.scenario_type.value,
            "scenario_status": scenario.status.value,
            "flighting_profile": (
                scenario.flighting_profile
            ),
            "date_month": scenario.date_month,
            "date_day": dates,
            "day_of_month": dates.day,
            "days_in_month": scenario.days,
            "flighting_weight": (
                flighting_weights
            ),
            "target_uncohorted_qs": (
                scenario.commercial_targets
                .target_uncohorted_qs
            ),
            "target_uncohorted_op": (
                scenario.commercial_targets
                .target_uncohorted_op
            ),
            "target_budget_media": (
                scenario.target_budget_media
            ),
        }
    )

    monthly_budget_values = {
        "budget_total": (
            scenario.budget_total
        ),
        "budget_media": (
            scenario.budget_media
        ),
        **(
            scenario.growth_budget_plan
            .channel_budgets
        ),
    }

    for (
        budget_column,
        monthly_value,
    ) in monthly_budget_values.items():
        daily_plan[
            f"daily_{budget_column}"
        ] = allocate_monthly_value_by_weights(
            monthly_value,
            flighting_weights,
        )

    daily_plan[
        "daily_target_uncohorted_qs"
    ] = allocate_monthly_value_by_weights(
        (
            scenario.commercial_targets
            .target_uncohorted_qs
        ),
        flighting_weights,
    )

    daily_plan[
        "daily_target_uncohorted_op"
    ] = allocate_monthly_value_by_weights(
        (
            scenario.commercial_targets
            .target_uncohorted_op
        ),
        flighting_weights,
    )

    validate_daily_scenario_plan(
        daily_plan=daily_plan,
        scenario=scenario,
        flighting_profiles=flighting_profiles,
    )

    return daily_plan


def validate_daily_scenario_plan(
    daily_plan: pd.DataFrame,
    scenario: Scenario,
    flighting_profiles: pd.DataFrame,
) -> None:
    """Validate one expanded daily scenario."""

    if daily_plan.empty:
        raise DailyPlanBuildError(f"daily plan is empty: {scenario.scenario_id}")

    if len(daily_plan) != scenario.days:
        raise DailyPlanBuildError(
            f"daily row count mismatch for {scenario.scenario_id}: "
            f"expected {scenario.days}, found {len(daily_plan)}"
        )

    if daily_plan["date_day"].duplicated().any():
        raise DailyPlanBuildError(
            f"duplicate date_day values found for {scenario.scenario_id}"
        )

    expected_weights = get_flighting_weights(
        flighting_profiles,
        profile_name=scenario.flighting_profile,
        days_in_month=scenario.days,
    )

    actual_weights = (
        daily_plan["flighting_weight"]
        .astype(float)
        .tolist()
    )

    if actual_weights != expected_weights:
        raise DailyPlanBuildError(
            f"flighting weights do not match profile "
            f"for {scenario.scenario_id}"
        )

    expected_values = {
        "daily_budget_total": scenario.budget_total,
        "daily_budget_media": scenario.budget_media,
        **{
            f"daily_{column}": value
            for column, value in scenario.growth_budget_plan.channel_budgets.items()
        },
        "daily_target_uncohorted_qs": scenario.commercial_targets.target_uncohorted_qs,
        "daily_target_uncohorted_op": scenario.commercial_targets.target_uncohorted_op,
    }

    errors: list[str] = []
    for column, expected_total in expected_values.items():
        actual_total = round(pd.to_numeric(daily_plan[column], errors="raise").sum(), 2)
        expected_total = round(float(expected_total), 2)
        if abs(actual_total - expected_total) > RECONCILIATION_TOLERANCE:
            errors.append(
                f"{column}: expected={expected_total}, actual={actual_total}"
            )

    if errors:
        raise DailyPlanBuildError(
            f"daily plan reconciliation failed for {scenario.scenario_id}:\n"
            + "\n".join(errors)
        )

def build_daily_plans(
    scenarios: Iterable[Scenario],
    flighting_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build and combine daily plans for multiple scenarios."""

    scenario_list = list(
        scenarios
    )

    if not scenario_list:
        raise DailyPlanBuildError(
            "no validated scenarios were supplied"
        )

    daily_frames = [
        build_daily_scenario_plan(
            scenario,
            flighting_profiles,
        )
        for scenario in scenario_list
    ]

    combined = (
        pd.concat(
            daily_frames,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            [
                "date_day",
                "scenario_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_keys = combined.duplicated(
        subset=[
            "scenario_id",
            "date_day",
        ],
        keep=False,
    )

    if duplicate_keys.any():
        duplicate_examples = (
            combined.loc[
                duplicate_keys,
                [
                    "scenario_id",
                    "date_day",
                ],
            ]
            .drop_duplicates()
            .head(20)
            .to_string(
                index=False
            )
        )

        raise DailyPlanBuildError(
            "duplicate scenario/date rows found:\n\n"
            + duplicate_examples
        )

    return combined


def build_monthly_summary(scenarios: Iterable[Scenario]) -> pd.DataFrame:
    """Build one flat summary row per Scenario."""

    return (
        pd.DataFrame(scenario.to_summary_dict() for scenario in scenarios)
        .sort_values(["date_month", "scenario_id"])
        .reset_index(drop=True)
    )


def save_planning_outputs(
    daily_plan: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    *,
    output_dir: Path = SCENARIO_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write scenario planning outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / DAILY_PLAN_OUTPUT_PATH.name
    monthly_path = output_dir / MONTHLY_SUMMARY_OUTPUT_PATH.name

    daily_output = daily_plan.copy()
    monthly_output = monthly_summary.copy()

    for dataframe in [daily_output, monthly_output]:
        for column in ["date_day", "date_month"]:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(dataframe[column]).dt.strftime(
                    "%Y-%m-%d"
                )

    daily_output.to_csv(daily_path, index=False)
    monthly_output.to_csv(monthly_path, index=False)
    return daily_path, monthly_path


def build_and_save_daily_plans() -> DailyPlanBuildResult:
    """Validate inputs, build all daily plans and save outputs."""

    validation_result = load_and_validate_planning_inputs()
    daily_plan = build_daily_plans(
        validation_result.scenarios,
        validation_result.flighting_profiles,
    )
    monthly_summary = build_monthly_summary(validation_result.scenarios)
    daily_path, monthly_path = save_planning_outputs(
        daily_plan,
        monthly_summary,
    )

    return DailyPlanBuildResult(
        daily_plan=daily_plan,
        monthly_summary=monthly_summary,
        daily_output_path=daily_path,
        monthly_output_path=monthly_path,
    )


def print_build_summary(result: DailyPlanBuildResult) -> None:
    """Print a compact daily-plan build report."""

    daily_plan = result.daily_plan
    monthly_summary = result.monthly_summary

    print("\n=====================================")
    print("==daily scenario plan")
    print(f"scenarios: {monthly_summary['scenario_id'].nunique():,}")
    print(f"daily rows: {len(daily_plan):,}")
    print(f"date min: {daily_plan['date_day'].min().date()}")
    print(f"date max: {daily_plan['date_day'].max().date()}")

    print("\n==monthly reconciliation")
    columns = [
        "scenario_id",
        "date_month",
        "target_budget_media",
        "budget_media",
        "media_target_variance",
        "budget_total",
        "allocated_budget_total",
        "budget_total_variance",
    ]
    print(monthly_summary[columns].to_string(index=False))

    print("\nsaved daily plan to:")
    print(result.daily_output_path)
    print("\nsaved monthly summary to:")
    print(result.monthly_output_path)
    print("\nall daily scenario plan checks passed")


def main() -> None:
    result = build_and_save_daily_plans()
    print_build_summary(result)


if __name__ == "__main__":
    try:
        main()
    except (PlanningInputValidationError, DailyPlanBuildError) as exc:
        raise SystemExit(f"daily scenario plan build failed:\n\n{exc}") from exc