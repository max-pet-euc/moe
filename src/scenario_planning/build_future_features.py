# src/scenario_planning/build_future_features.py
# FUTURE_SCENARIO_FEATURE_VERSION = "1.0-20260804_1409"

"""
Build a future scenario feature dataset from the validated daily plan.

Inputs:
- data/engineered/data_features.csv
- data/outputs/scenarios/scenario_daily_plan.csv

Outputs:
- data/outputs/scenarios/scenario_future_features.csv
- data/outputs/scenarios/scenario_feature_assumptions.csv

Approach:
1. Use the canonical feature dataset as the date/calendar template.
2. Overlay the planned daily aggregate budgets.
3. Split Search and Meta using recent historical spend shares.
4. Convert detailed digital spend into expected impressions, clicks,
   platform QS and platform CP using recent historical efficiency.
5. Carry forward remaining non-controllable inputs using recent
   day-of-week averages.
6. Recalculate the standard engineered marketing features.

This creates model-ready scenario inputs. It does not yet generate
QS or OP predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ENGINEERED_DIR
from src.feature_pipeline.engineered_features import (
    CHANNEL_METRIC_GROUPS,
)
from src.feature_pipeline.feature_engineering import (
    add_engineered_features,
    clean_numeric_series,
)

from .scenario_feature_settings import (
    ASSUMPTION_TOLERANCE,
    DEFAULT_SHARE_WHEN_NO_HISTORY,
    DIRECT_DIGITAL_SPEND_MAPPINGS,
    DIRECT_MEDIA_BUDGET_MAPPINGS,
    EXCLUDED_FUTURE_COLUMNS,
    EXCLUDED_FUTURE_PREFIXES,
    HISTORICAL_LOOKBACK_DAYS,
    META_DIGITAL_SPEND_COLUMNS,
    MINIMUM_HISTORY_DAYS,
    SEARCH_DIGITAL_SPEND_COLUMNS,
    SEARCH_MEDIA_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "scenarios"
)

SCENARIO_DAILY_PLAN_PATH = (
    SCENARIO_OUTPUT_DIR
    / "scenario_daily_plan.csv"
)

HISTORICAL_FEATURE_PATH = (
    ENGINEERED_DIR
    / "data_features.csv"
)

FUTURE_FEATURE_OUTPUT_PATH = (
    SCENARIO_OUTPUT_DIR
    / "scenario_future_features.csv"
)

ASSUMPTION_OUTPUT_PATH = (
    SCENARIO_OUTPUT_DIR
    / "scenario_feature_assumptions.csv"
)


class FutureFeatureBuildError(
    ValueError
):
    """Raised when scenario features cannot be built safely."""


@dataclass(frozen=True)
class FutureFeatureBuildResult:
    """Scenario feature data and its supporting assumptions."""

    future_features: pd.DataFrame
    assumptions: pd.DataFrame
    feature_output_path: Path
    assumption_output_path: Path


def load_csv_with_dates(
    path: Path,
) -> pd.DataFrame:
    """Load a CSV and parse its date columns."""

    if not path.exists():
        raise FutureFeatureBuildError(
            f"required file not found: {path}"
        )

    output = pd.read_csv(
        path,
        low_memory=False,
    )

    for column in [
        "date_day",
        "date_month",
        "date_week",
    ]:
        if column in output.columns:
            output[column] = pd.to_datetime(
                output[column],
                errors="raise",
            )

    return output


def get_actual_cutoff_date(
    historical_features: pd.DataFrame,
) -> pd.Timestamp:
    """
    Return the latest date containing an observed business outcome.

    OP is preferred, then CP, then QS.
    """

    candidate_columns = [
        "uncohorted_op",
        "uncohorted_cp",
        "uncohorted_qs",
    ]

    for column in candidate_columns:
        if column not in historical_features.columns:
            continue

        observed = historical_features.loc[
            clean_numeric_series(
                historical_features[column]
            ).notna(),
            "date_day",
        ]

        if not observed.empty:
            return pd.Timestamp(
                observed.max()
            ).normalize()

    raise FutureFeatureBuildError(
        "could not identify an actual-data cutoff from "
        "uncohorted OP, CP or QS"
    )


def get_reference_history(
    historical_features: pd.DataFrame,
    *,
    scenario_start_date: pd.Timestamp,
    actual_cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return the recent actual window used for assumptions."""

    reference_end = min(
        scenario_start_date
        - pd.Timedelta(days=1),
        actual_cutoff_date,
    )

    reference_start = (
        reference_end
        - pd.Timedelta(
            days=HISTORICAL_LOOKBACK_DAYS - 1
        )
    )

    history = (
        historical_features.loc[
            historical_features[
                "date_day"
            ].between(
                reference_start,
                reference_end,
                inclusive="both",
            )
        ]
        .copy()
        .sort_values("date_day")
        .reset_index(drop=True)
    )

    if len(history) < MINIMUM_HISTORY_DAYS:
        raise FutureFeatureBuildError(
            "insufficient actual history for scenario month "
            f"{scenario_start_date:%Y-%m-%d}; "
            f"reference window={reference_start:%Y-%m-%d} "
            f"to {reference_end:%Y-%m-%d}; "
            f"rows={len(history)}"
        )

    return history


def normalise_shares(
    values: pd.Series,
) -> pd.Series:
    """Convert positive values into shares summing exactly to one."""

    numeric = (
        clean_numeric_series(values)
        .fillna(0)
        .clip(lower=0)
    )

    total = float(
        numeric.sum()
    )

    if total <= 0:
        return pd.Series(
            DEFAULT_SHARE_WHEN_NO_HISTORY,
            index=values.index,
            dtype="float64",
        )

    shares = numeric / total

    difference = (
        1.0
        - float(shares.sum())
    )

    shares.iloc[-1] = (
        shares.iloc[-1]
        + difference
    )

    return shares


def calculate_spend_shares(
    history: pd.DataFrame,
    columns: list[str],
) -> dict[str, float]:
    """Calculate recent spend shares for configured columns."""

    totals = pd.Series(
        {
            column: (
                clean_numeric_series(
                    history[column]
                ).sum()
                if column in history.columns
                else 0.0
            )
            for column in columns
        },
        dtype="float64",
    )

    shares = normalise_shares(
        totals
    )

    if float(shares.sum()) <= 0:
        equal_share = (
            1.0 / len(columns)
        )

        return {
            column: equal_share
            for column in columns
        }

    return {
        column: float(
            shares[column]
        )
        for column in columns
    }


def apply_split_budget(
    frame: pd.DataFrame,
    *,
    source_column: str,
    destination_shares: dict[str, float],
) -> None:
    """Split one planned budget across detailed feature columns."""

    if source_column not in frame.columns:
        raise FutureFeatureBuildError(
            f"daily plan is missing {source_column}"
        )

    source_values = clean_numeric_series(
        frame[source_column]
    ).fillna(0)

    for (
        destination_column,
        share,
    ) in destination_shares.items():
        frame[destination_column] = (
            source_values
            * float(share)
        )


def calculate_rate(
    history: pd.DataFrame,
    *,
    numerator_column: str,
    denominator_column: str,
) -> float:
    """Calculate a stable aggregate numerator-per-denominator rate."""

    if (
        numerator_column not in history.columns
        or denominator_column not in history.columns
    ):
        return 0.0

    numerator = (
        clean_numeric_series(
            history[numerator_column]
        )
        .fillna(0)
        .clip(lower=0)
        .sum()
    )

    denominator = (
        clean_numeric_series(
            history[denominator_column]
        )
        .fillna(0)
        .clip(lower=0)
        .sum()
    )

    if denominator <= 0:
        return 0.0

    return float(
        numerator / denominator
    )


def add_expected_platform_metrics(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    *,
    scenario_id: str,
    scenario_month: pd.Timestamp,
) -> list[dict[str, object]]:
    """
    Derive future impressions, clicks, QS and CP from planned spend.

    Rates use aggregate recent outcomes per pound:
    - impressions / spend
    - clicks / spend
    - QS / spend
    - CP / spend
    """

    assumption_rows: list[
        dict[str, object]
    ] = []

    for (
        group_name,
        columns,
    ) in CHANNEL_METRIC_GROUPS.items():
        spend_column = columns.get(
            "cost"
        )

        if (
            not spend_column
            or spend_column not in frame.columns
        ):
            continue

        for metric_key in [
            "impressions",
            "clicks",
            "qs",
            "cp",
        ]:
            metric_column = columns.get(
                metric_key
            )

            if not metric_column:
                continue

            rate = calculate_rate(
                history,
                numerator_column=metric_column,
                denominator_column=spend_column,
            )

            frame[metric_column] = (
                clean_numeric_series(
                    frame[spend_column]
                )
                .fillna(0)
                * rate
            )

            assumption_rows.append(
                {
                    "scenario_id": scenario_id,
                    "date_month": (
                        scenario_month
                    ),
                    "assumption_type": (
                        "platform_metric_rate"
                    ),
                    "feature_group": (
                        group_name
                    ),
                    "source_feature": (
                        spend_column
                    ),
                    "output_feature": (
                        metric_column
                    ),
                    "assumption_value": rate,
                    "assumption_unit": (
                        f"{metric_key}_per_currency"
                    ),
                }
            )

    return assumption_rows


def apply_recent_baselines(
    scenario_frame: pd.DataFrame,
    history: pd.DataFrame,
    *,
    protected_columns: set[str],
) -> list[dict[str, object]]:
    """
    Fill remaining numeric scenario inputs using recent weekday means.

    This is used for non-planned contextual features such as external
    metrics and calendar-linked demand variables.
    """

    assumption_rows: list[
        dict[str, object]
    ] = []

    history_with_weekday = (
        history.copy()
    )

    history_with_weekday[
        "_weekday"
    ] = (
        history_with_weekday[
            "date_day"
        ].dt.dayofweek
    )

    scenario_weekday = (
        scenario_frame[
            "date_day"
        ].dt.dayofweek
    )

    numeric_candidates = [
        column
        for column in history.columns
        if column not in protected_columns
        and column != "date_day"
        and not column.startswith(
            EXCLUDED_FUTURE_PREFIXES
        )
        and column
        not in EXCLUDED_FUTURE_COLUMNS
    ]

    for column in numeric_candidates:
        historical_numeric = (
            clean_numeric_series(
                history[column]
            )
        )

        if not historical_numeric.notna().any():
            continue

        if column not in scenario_frame.columns:
            scenario_frame[column] = np.nan

        existing_numeric = (
            clean_numeric_series(
                scenario_frame[column]
            )
        )

        missing_mask = (
            existing_numeric.isna()
        )

        if not missing_mask.any():
            scenario_frame[column] = (
                existing_numeric
            )
            continue

        weekday_means = (
            pd.DataFrame(
                {
                    "_weekday": (
                        history_with_weekday[
                            "_weekday"
                        ]
                    ),
                    "_value": (
                        historical_numeric
                    ),
                }
            )
            .groupby(
                "_weekday"
            )["_value"]
            .mean()
        )

        overall_mean = float(
            historical_numeric.mean()
        )

        fill_values = (
            scenario_weekday.map(
                weekday_means
            )
            .fillna(
                overall_mean
            )
        )

        scenario_frame.loc[
            missing_mask,
            column,
        ] = fill_values.loc[
            missing_mask
        ]

        assumption_rows.append(
            {
                "assumption_type": (
                    "recent_weekday_baseline"
                ),
                "feature_group": (
                    "context"
                ),
                "source_feature": column,
                "output_feature": column,
                "assumption_value": (
                    overall_mean
                ),
                "assumption_unit": (
                    "historical_mean"
                ),
            }
        )

    return assumption_rows


def build_month_scenario_features(
    daily_plan_month: pd.DataFrame,
    historical_features: pd.DataFrame,
    *,
    actual_cutoff_date: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
]:
    """Build model inputs for one monthly Scenario."""

    if daily_plan_month.empty:
        raise FutureFeatureBuildError(
            "scenario month is empty"
        )

    scenario_id = str(
        daily_plan_month[
            "scenario_id"
        ].iloc[0]
    )

    scenario_month = pd.Timestamp(
        daily_plan_month[
            "date_month"
        ].iloc[0]
    ).normalize()

    history = get_reference_history(
        historical_features,
        scenario_start_date=(
            scenario_month
        ),
        actual_cutoff_date=(
            actual_cutoff_date
        ),
    )

    date_template = (
        historical_features.loc[
            historical_features[
                "date_day"
            ].isin(
                daily_plan_month[
                    "date_day"
                ]
            )
        ]
        .copy()
    )

    if (
        len(date_template)
        != len(daily_plan_month)
    ):
        missing_dates = sorted(
            set(
                daily_plan_month[
                    "date_day"
                ]
            )
            - set(
                date_template[
                    "date_day"
                ]
            )
        )

        raise FutureFeatureBuildError(
            "canonical feature dataset is missing "
            f"scenario dates for {scenario_id}: "
            f"{missing_dates}"
        )

    scenario_frame = (
        daily_plan_month
        .merge(
            date_template,
            on="date_day",
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_historical",
            ),
        )
        .sort_values(
            "date_day"
        )
        .reset_index(
            drop=True
        )
    )

    historical_duplicate_columns = [
        column
        for column in scenario_frame.columns
        if column.endswith(
            "_historical"
        )
    ]

    scenario_frame = (
        scenario_frame.drop(
            columns=(
                historical_duplicate_columns
            ),
            errors="ignore",
        )
    )

    columns_to_drop = [
        column
        for column in scenario_frame.columns
        if column.startswith(
            EXCLUDED_FUTURE_PREFIXES
        )
    ]

    scenario_frame = (
        scenario_frame.drop(
            columns=columns_to_drop,
            errors="ignore",
        )
    )

    assumption_rows: list[
        dict[str, object]
    ] = []

    # -----------------------------------------------------
    # Direct aggregate media-input mappings
    # -----------------------------------------------------

    for (
        plan_column,
        feature_column,
    ) in DIRECT_MEDIA_BUDGET_MAPPINGS.items():
        if plan_column not in scenario_frame.columns:
            continue

        scenario_frame[
            feature_column
        ] = clean_numeric_series(
            scenario_frame[
                plan_column
            ]
        ).fillna(0)

    # -----------------------------------------------------
    # Search split
    # -----------------------------------------------------

    search_shares = calculate_spend_shares(
        history,
        SEARCH_MEDIA_COLUMNS,
    )

    apply_split_budget(
        scenario_frame,
        source_column=(
            "daily_budget_search"
        ),
        destination_shares=(
            search_shares
        ),
    )

    for (
        media_column,
        digital_column,
    ) in (
        SEARCH_DIGITAL_SPEND_COLUMNS.items()
    ):
        scenario_frame[
            digital_column
        ] = clean_numeric_series(
            scenario_frame[
                media_column
            ]
        ).fillna(0)

    for (
        destination,
        share,
    ) in search_shares.items():
        assumption_rows.append(
            {
                "scenario_id": scenario_id,
                "date_month": scenario_month,
                "assumption_type": (
                    "historical_spend_share"
                ),
                "feature_group": "search",
                "source_feature": (
                    "daily_budget_search"
                ),
                "output_feature": destination,
                "assumption_value": share,
                "assumption_unit": "share",
            }
        )

    # -----------------------------------------------------
    # Meta split
    # -----------------------------------------------------

    meta_shares = calculate_spend_shares(
        history,
        META_DIGITAL_SPEND_COLUMNS,
    )

    apply_split_budget(
        scenario_frame,
        source_column=(
            "daily_budget_meta"
        ),
        destination_shares=(
            meta_shares
        ),
    )

    for (
        destination,
        share,
    ) in meta_shares.items():
        assumption_rows.append(
            {
                "scenario_id": scenario_id,
                "date_month": scenario_month,
                "assumption_type": (
                    "historical_spend_share"
                ),
                "feature_group": "meta",
                "source_feature": (
                    "daily_budget_meta"
                ),
                "output_feature": destination,
                "assumption_value": share,
                "assumption_unit": "share",
            }
        )

    # -----------------------------------------------------
    # Direct digital spend
    # -----------------------------------------------------

    for (
        plan_column,
        digital_column,
    ) in (
        DIRECT_DIGITAL_SPEND_MAPPINGS.items()
    ):
        if plan_column not in scenario_frame.columns:
            continue

        scenario_frame[
            digital_column
        ] = clean_numeric_series(
            scenario_frame[
                plan_column
            ]
        ).fillna(0)

    # -----------------------------------------------------
    # Expected digital metrics from planned spend
    # -----------------------------------------------------

    assumption_rows.extend(
        add_expected_platform_metrics(
            scenario_frame,
            history,
            scenario_id=scenario_id,
            scenario_month=scenario_month,
        )
    )

    # -----------------------------------------------------
    # Remaining recent contextual assumptions
    # -----------------------------------------------------

    protected_columns = {
        "date_day",
        "date_month",
        "date_week",
        *[
            column
            for column in scenario_frame.columns
            if column.startswith(
                "daily_"
            )
            or column.startswith(
                "target_"
            )
            or column.startswith(
                "scenario_"
            )
        ],
        *DIRECT_MEDIA_BUDGET_MAPPINGS.values(),
        *SEARCH_MEDIA_COLUMNS,
        *SEARCH_DIGITAL_SPEND_COLUMNS.values(),
        *META_DIGITAL_SPEND_COLUMNS,
        *DIRECT_DIGITAL_SPEND_MAPPINGS.values(),
        *[
            metric_column
            for columns
            in CHANNEL_METRIC_GROUPS.values()
            for metric_column
            in columns.values()
            if metric_column
        ],
    }

    baseline_rows = (
        apply_recent_baselines(
            scenario_frame,
            history,
            protected_columns=(
                protected_columns
            ),
        )
    )

    for row in baseline_rows:
        row.update(
            {
                "scenario_id": scenario_id,
                "date_month": (
                    scenario_month
                ),
            }
        )

    assumption_rows.extend(
        baseline_rows
    )

    # -----------------------------------------------------
    # Recalculate engineered features
    # -----------------------------------------------------

    scenario_frame = (
        add_engineered_features(
            scenario_frame
        )
    )

    # Remove intentionally excluded raw totals if present.
    scenario_frame = (
        scenario_frame.drop(
            columns=[
                column
                for column in EXCLUDED_FUTURE_COLUMNS
                if column
                in scenario_frame.columns
            ],
            errors="ignore",
        )
    )

    scenario_frame[
        "scenario_reference_start"
    ] = history[
        "date_day"
    ].min()

    scenario_frame[
        "scenario_reference_end"
    ] = history[
        "date_day"
    ].max()

    return (
        scenario_frame,
        assumption_rows,
    )


def validate_future_features(
    future_features: pd.DataFrame,
) -> None:
    """Run core validation on the scenario feature output."""

    if future_features.empty:
        raise FutureFeatureBuildError(
            "future scenario feature dataset is empty"
        )

    duplicate_rows = (
        future_features.duplicated(
            subset=[
                "scenario_id",
                "date_day",
            ],
            keep=False,
        )
    )

    if duplicate_rows.any():
        raise FutureFeatureBuildError(
            "future scenario feature dataset contains "
            "duplicate scenario/date rows"
        )

    attribution_columns = [
        column
        for column in future_features.columns
        if column.startswith(
            "attribution_"
        )
    ]

    if attribution_columns:
        raise FutureFeatureBuildError(
            "attribution columns remain in future features: "
            + ", ".join(
                attribution_columns[:20]
            )
        )

    planned_budget_columns = [
        column
        for column in future_features.columns
        if column.startswith(
            "daily_budget_"
        )
    ]

    if not planned_budget_columns:
        raise FutureFeatureBuildError(
            "future scenario features contain no "
            "daily budget columns"
        )

    numeric_columns = (
        future_features
        .select_dtypes(
            include="number"
        )
    )

    numeric_array = (
        numeric_columns.to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
    )

    if np.isinf(
        numeric_array
    ).any():
        raise FutureFeatureBuildError(
            "future scenario features contain "
            "infinite numeric values"
        )


def build_future_scenario_features(
) -> FutureFeatureBuildResult:
    """Build and save all scenario feature rows."""

    historical_features = (
        load_csv_with_dates(
            HISTORICAL_FEATURE_PATH
        )
    )

    daily_plan = load_csv_with_dates(
        SCENARIO_DAILY_PLAN_PATH
    )

    actual_cutoff_date = (
        get_actual_cutoff_date(
            historical_features
        )
    )

    monthly_frames: list[
        pd.DataFrame
    ] = []

    assumption_rows: list[
        dict[str, object]
    ] = []

    for (
        _,
        daily_plan_month,
    ) in daily_plan.groupby(
        "scenario_id",
        sort=True,
    ):
        (
            month_features,
            month_assumptions,
        ) = build_month_scenario_features(
            daily_plan_month,
            historical_features,
            actual_cutoff_date=(
                actual_cutoff_date
            ),
        )

        monthly_frames.append(
            month_features
        )

        assumption_rows.extend(
            month_assumptions
        )

    future_features = (
        pd.concat(
            monthly_frames,
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

    assumptions = (
        pd.DataFrame(
            assumption_rows
        )
        .sort_values(
            [
                "date_month",
                "scenario_id",
                "assumption_type",
                "feature_group",
                "output_feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_future_features(
        future_features
    )

    SCENARIO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_features = (
        future_features.copy()
    )

    output_assumptions = (
        assumptions.copy()
    )

    for frame in [
        output_features,
        output_assumptions,
    ]:
        for column in [
            "date_day",
            "date_month",
            "date_week",
            "scenario_reference_start",
            "scenario_reference_end",
        ]:
            if column in frame.columns:
                frame[column] = (
                    pd.to_datetime(
                        frame[column]
                    )
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                )

    output_features.to_csv(
        FUTURE_FEATURE_OUTPUT_PATH,
        index=False,
    )

    output_assumptions.to_csv(
        ASSUMPTION_OUTPUT_PATH,
        index=False,
    )

    return FutureFeatureBuildResult(
        future_features=(
            future_features
        ),
        assumptions=assumptions,
        feature_output_path=(
            FUTURE_FEATURE_OUTPUT_PATH
        ),
        assumption_output_path=(
            ASSUMPTION_OUTPUT_PATH
        ),
    )


def print_build_summary(
    result: FutureFeatureBuildResult,
) -> None:
    """Print a compact scenario-feature report."""

    features = result.future_features
    assumptions = result.assumptions

    print(
        "\n====================================="
    )
    print(
        "==future scenario features"
    )
    print(
        f"scenarios: "
        f"{features['scenario_id'].nunique():,}"
    )
    print(
        f"rows: {len(features):,}"
    )
    print(
        f"columns: {len(features.columns):,}"
    )
    print(
        f"date min: "
        f"{features['date_day'].min().date()}"
    )
    print(
        f"date max: "
        f"{features['date_day'].max().date()}"
    )

    print(
        "\n==assumptions"
    )
    print(
        assumptions[
            "assumption_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nsaved future features to:"
    )
    print(
        result.feature_output_path
    )

    print(
        "\nsaved assumptions to:"
    )
    print(
        result.assumption_output_path
    )

    print(
        "\nall future scenario feature checks passed"
    )


def main() -> None:
    """Command-line entry point."""

    result = (
        build_future_scenario_features()
    )

    print_build_summary(
        result
    )


if __name__ == "__main__":
    try:
        main()
    except FutureFeatureBuildError as exc:
        raise SystemExit(
            "future scenario feature build failed:\n\n"
            f"{exc}"
        ) from exc
