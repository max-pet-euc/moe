"""
Explain movement between two configured modelled periods.

Report contract version 2.2 adds:
- configurable current and previous date ranges;
- period-level actual and prediction aggregation;
- SHAP aggregation across every row in both periods;
- rolling like-for-like historical movement ranking;
- average daily feature values for descriptive context.

"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class ExplanationResult:
    previous_period: object
    current_period: object

    previous_actual: float
    current_actual: float

    previous_predicted: float
    current_predicted: float

    expected_value: float
    prediction_vs_baseline: float

    actual_change: float
    actual_change_pct: float
    predicted_change: float
    actual_vs_expected_movement: float
    prediction_error: float
    movement_scaling_factor: float
    movement_percentile: float
    movement_rank: int
    movement_history_count: int

    contribution_df: pd.DataFrame
    summary: str

    @property
    def movement_accuracy_ratio(self) -> float:
        """How much of the predicted movement was observed in reality."""

        if self.predicted_change == 0:
            return np.nan

        return self.actual_change / self.predicted_change

    @property
    def surprise_direction(self) -> str:
        """Human-readable direction of the actual-versus-expected residual."""

        if self.actual_vs_expected_movement > 0:
            return "better_than_expected"

        if self.actual_vs_expected_movement < 0:
            return "worse_than_expected"

        return "matched_expectation"

    @property
    def unexplained_change(self) -> float:
        """Backward-compatible alias."""

        return self.actual_vs_expected_movement


def _format_period(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """Format a reporting period for use in the HTML narrative."""

    return (
        f"{start_date:%d %b %Y}"
        f" – "
        f"{end_date:%d %b %Y}"
    )


def _movement_text(
    change: float,
) -> str:
    """Return a grammatically complete description of a numeric movement."""

    if change > 0:
        return f"rose by {abs(change):,.1f}"

    if change < 0:
        return f"fell by {abs(change):,.1f}"

    return "was unchanged"


def build_summary(
    model_stage: str,
    previous_period,
    current_period,
    previous_actual: float,
    current_actual: float,
    actual_change: float,
    previous_predicted: float,
    current_predicted: float,
    predicted_change: float,
    actual_vs_expected_movement: float,
    contributions: pd.DataFrame,
) -> str:
    """Build the report's plain-English period comparison narrative."""

    stage_label = model_stage.upper()

    actual_sentence = (
        f"Actual {stage_label} "
        f"{_movement_text(actual_change)}, "
        f"from {previous_actual:,.1f} in "
        f"{previous_period} to "
        f"{current_actual:,.1f} in "
        f"{current_period}."
    )

    prediction_sentence = (
        f"The model expected {stage_label} "
        f"to move from {previous_predicted:,.1f} to "
        f"{current_predicted:,.1f}, "
        f"a change of {predicted_change:+,.1f}."
    )

    if actual_vs_expected_movement > 0:
        surprise_sentence = (
            f"Actual movement was "
            f"{abs(actual_vs_expected_movement):,.1f} "
            "better than expected."
        )
    elif actual_vs_expected_movement < 0:
        surprise_sentence = (
            f"Actual movement was "
            f"{abs(actual_vs_expected_movement):,.1f} "
            "worse than expected."
        )
    else:
        surprise_sentence = (
            "Actual movement matched the model's expectation."
        )

    positive_contributors = (
        contributions
        .loc[
            contributions[
                "model_expected_contribution"
            ] > 0
        ]
        .sort_values(
            "model_expected_contribution",
            ascending=False,
        )
        .head(3)
    )

    negative_contributors = (
        contributions
        .loc[
            contributions[
                "model_expected_contribution"
            ] < 0
        ]
        .sort_values(
            "model_expected_contribution",
            ascending=True,
        )
        .head(3)
    )

    driver_sentences = []

    if not positive_contributors.empty:
        positive_text = ", ".join(
            (
                f"{row.feature} "
                f"({row.model_expected_contribution:+,.1f})"
            )
            for row in positive_contributors.itertuples()
        )

        driver_sentences.append(
            f"The main upward model drivers were {positive_text}."
        )

    if not negative_contributors.empty:
        negative_text = ", ".join(
            (
                f"{row.feature} "
                f"({row.model_expected_contribution:+,.1f})"
            )
            for row in negative_contributors.itertuples()
        )

        driver_sentences.append(
            f"The main downward model drivers were {negative_text}."
        )

    return " ".join(
        [
            actual_sentence,
            prediction_sentence,
            surprise_sentence,
            *driver_sentences,
        ]
    )


def aggregate_values(
    values,
    aggregation: str,
) -> float:
    """Aggregate a numeric period using the configured reporting method."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric_values.isna().all():
        return np.nan

    if aggregation == "sum":
        return float(
            numeric_values.sum(
                min_count=1,
            )
        )

    if aggregation == "mean":
        return float(
            numeric_values.mean()
        )

    raise ValueError(
        f"unsupported aggregation: {aggregation}"
    )


def _aggregate_shap_values(
    shap_values: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    """Aggregate row-level SHAP values into a period-level explanation."""

    if aggregation == "sum":
        return shap_values.sum(
            axis=0
        )

    if aggregation == "mean":
        return shap_values.mean(
            axis=0
        )

    raise ValueError(
        f"unsupported aggregation: {aggregation}"
    )


def _aggregate_base_values(
    base_values: np.ndarray,
    aggregation: str,
) -> float:
    """Aggregate row-level SHAP base values consistently with predictions."""

    flattened_values = np.asarray(
        base_values
    ).reshape(-1)

    if aggregation == "sum":
        return float(
            flattened_values.sum()
        )

    if aggregation == "mean":
        return float(
            flattened_values.mean()
        )

    raise ValueError(
        f"unsupported aggregation: {aggregation}"
    )


def _historical_period_movements(
    prediction_data: pd.DataFrame,
    actual_column: str,
    date_column: str,
    period_days: int,
    aggregation: str,
) -> pd.Series:
    """
    Create like-for-like historical movements.

    For a seven-day report, this compares each rolling seven-day value with
    the immediately preceding seven-day value.
    """

    actual_series = (
        prediction_data
        .sort_values(
            date_column
        )
        .set_index(
            date_column
        )[
            actual_column
        ]
        .pipe(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if aggregation == "sum":
        historical_period_values = (
            actual_series
            .rolling(
                window=period_days,
                min_periods=period_days,
            )
            .sum()
        )
    elif aggregation == "mean":
        historical_period_values = (
            actual_series
            .rolling(
                window=period_days,
                min_periods=period_days,
            )
            .mean()
        )
    else:
        raise ValueError(
            f"unsupported aggregation: {aggregation}"
        )

    return (
        historical_period_values
        - historical_period_values.shift(
            period_days
        )
    ).dropna().abs()


def explain_period_change(
    model,
    x_reference: pd.DataFrame,
    x_explain: pd.DataFrame,
    prediction_df: pd.DataFrame,
    model_stage: str,
    model_grain: str,
    current_start_date,
    current_end_date,
    previous_start_date,
    previous_end_date,
    aggregation: str = "sum",
    date_column: str = "date_day",
    background_rows: int = 100,
    random_state: int = 42,
) -> ExplanationResult:
    """
    Explain the modelled movement between two configured reporting periods.

    The model remains daily. Actuals, predictions, SHAP values and SHAP base
    values are aggregated across every daily row in each reporting period.
    """

    del model_grain  # Retained in the public signature for compatibility.

    valid_aggregations = {
        "sum",
        "mean",
    }

    if aggregation not in valid_aggregations:
        raise ValueError(
            "aggregation must be one of: "
            f"{sorted(valid_aggregations)}"
        )

    if x_explain.empty:
        raise ValueError(
            "x_explain cannot be empty"
        )

    actual_column = (
        f"actual_{model_stage}"
    )

    predicted_column = (
        f"predicted_{model_stage}"
    )

    required_prediction_columns = {
        date_column,
        actual_column,
        predicted_column,
    }

    missing_columns = (
        required_prediction_columns
        - set(
            prediction_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "prediction dataframe is missing: "
            f"{sorted(missing_columns)}"
        )

    prediction_data = (
        prediction_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    feature_data = (
        x_explain
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(prediction_data) != len(feature_data):
        raise ValueError(
            "prediction_df and x_explain must "
            "contain the same number of rows"
        )

    prediction_data[
        date_column
    ] = pd.to_datetime(
        prediction_data[
            date_column
        ],
        errors="coerce",
    ).dt.normalize()

    if prediction_data[
        date_column
    ].isna().any():
        raise ValueError(
            f"{date_column} contains values "
            "that could not be converted to dates"
        )

    current_start_date = pd.Timestamp(
        current_start_date
    ).normalize()

    current_end_date = pd.Timestamp(
        current_end_date
    ).normalize()

    previous_start_date = pd.Timestamp(
        previous_start_date
    ).normalize()

    previous_end_date = pd.Timestamp(
        previous_end_date
    ).normalize()

    if current_start_date > current_end_date:
        raise ValueError(
            "current_start_date must be on or before "
            "current_end_date"
        )

    if previous_start_date > previous_end_date:
        raise ValueError(
            "previous_start_date must be on or before "
            "previous_end_date"
        )

    current_period_days = (
        current_end_date
        - current_start_date
    ).days + 1

    previous_period_days = (
        previous_end_date
        - previous_start_date
    ).days + 1

    if current_period_days != previous_period_days:
        raise ValueError(
            "current and previous reporting periods "
            "must contain the same number of days"
        )

    current_mask = (
        prediction_data[
            date_column
        ]
        .between(
            current_start_date,
            current_end_date,
            inclusive="both",
        )
    )

    previous_mask = (
        prediction_data[
            date_column
        ]
        .between(
            previous_start_date,
            previous_end_date,
            inclusive="both",
        )
    )

    if not current_mask.any():
        raise ValueError(
            "no prediction rows found for the "
            "current reporting period"
        )

    if not previous_mask.any():
        raise ValueError(
            "no prediction rows found for the "
            "previous reporting period"
        )

    current_positions = np.flatnonzero(
        current_mask.to_numpy()
    )

    previous_positions = np.flatnonzero(
        previous_mask.to_numpy()
    )

    current_features = (
        feature_data
        .iloc[
            current_positions
        ]
        .copy()
    )

    previous_features = (
        feature_data
        .iloc[
            previous_positions
        ]
        .copy()
    )

    if len(current_features) != current_period_days:
        raise ValueError(
            "current reporting period contains "
            f"{len(current_features)} rows but "
            f"{current_period_days} daily rows were expected"
        )

    if len(previous_features) != previous_period_days:
        raise ValueError(
            "previous reporting period contains "
            f"{len(previous_features)} rows but "
            f"{previous_period_days} daily rows were expected"
        )

    explain_features = pd.concat(
        [
            previous_features,
            current_features,
        ],
        ignore_index=True,
    )

    reference_data = (
        x_reference
        .copy()
    )

    if reference_data.empty:
        raise ValueError(
            "x_reference cannot be empty"
        )

    if len(reference_data) > background_rows:
        reference_data = (
            reference_data
            .sample(
                n=background_rows,
                random_state=random_state,
            )
        )

    reference_data = (
        reference_data
        .reset_index(
            drop=True
        )
    )

    explainer = shap.Explainer(
        model.predict,
        reference_data,
        feature_names=list(
            feature_data.columns
        ),
        algorithm="permutation",
    )

    previous_row_count = len(
        previous_features
    )

    minimum_evaluations = (
        2
        * len(
            feature_data.columns
        )
        + 1
    )

    shap_values = explainer(
        explain_features,
        max_evals=max(
            minimum_evaluations,
            500,
        ),
    )

    all_shap_values = np.asarray(
        shap_values.values
    )

    previous_shap_values = (
        all_shap_values[
            :previous_row_count
        ]
    )

    current_shap_values = (
        all_shap_values[
            previous_row_count:
        ]
    )

    previous_shap = _aggregate_shap_values(
        previous_shap_values,
        aggregation=aggregation,
    )

    current_shap = _aggregate_shap_values(
        current_shap_values,
        aggregation=aggregation,
    )

    model_expected_contribution = (
        current_shap
        - previous_shap
    )

    all_base_values = np.asarray(
        shap_values.base_values
    ).reshape(-1)

    previous_base_values = (
        all_base_values[
            :previous_row_count
        ]
    )

    current_base_values = (
        all_base_values[
            previous_row_count:
        ]
    )

    previous_expected_value = _aggregate_base_values(
        previous_base_values,
        aggregation=aggregation,
    )

    expected_value = _aggregate_base_values(
        current_base_values,
        aggregation=aggregation,
    )

    previous_prediction_data = (
        prediction_data
        .loc[
            previous_mask
        ]
        .copy()
    )

    current_prediction_data = (
        prediction_data
        .loc[
            current_mask
        ]
        .copy()
    )

    previous_actual = aggregate_values(
        previous_prediction_data[
            actual_column
        ],
        aggregation=aggregation,
    )

    current_actual = aggregate_values(
        current_prediction_data[
            actual_column
        ],
        aggregation=aggregation,
    )

    previous_predicted = aggregate_values(
        previous_prediction_data[
            predicted_column
        ],
        aggregation=aggregation,
    )

    current_predicted = aggregate_values(
        current_prediction_data[
            predicted_column
        ],
        aggregation=aggregation,
    )

    actual_change = (
        current_actual
        - previous_actual
    )

    predicted_change = (
        current_predicted
        - previous_predicted
    )

    actual_vs_expected_movement = (
        actual_change
        - predicted_change
    )

    prediction_error = (
        current_actual
        - current_predicted
    )

    prediction_vs_baseline = (
        current_predicted
        - expected_value
    )

    if previous_actual == 0 or pd.isna(previous_actual):
        actual_change_pct = np.nan
    else:
        actual_change_pct = (
            actual_change
            / previous_actual
            * 100
        )

    if predicted_change == 0 or pd.isna(predicted_change):
        movement_scaling_factor = np.nan
    else:
        movement_scaling_factor = (
            actual_change
            / predicted_change
        )

    if pd.isna(movement_scaling_factor):
        proportional_actual_contribution = np.full(
            shape=len(
                model_expected_contribution
            ),
            fill_value=np.nan,
            dtype=float,
        )
    else:
        proportional_actual_contribution = (
            model_expected_contribution
            * movement_scaling_factor
        )

    # Feature values are displayed as average daily values even when the
    # target and SHAP effects are summed across the reporting period.
    previous_feature_values = (
        previous_features
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .mean(
            axis=0
        )
    )

    current_feature_values = (
        current_features
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .mean(
            axis=0
        )
    )

    # Percentiles are descriptive context only, not causal evidence.
    current_feature_percentiles = []

    for feature in feature_data.columns:
        reference_values = pd.to_numeric(
            reference_data[
                feature
            ],
            errors="coerce",
        ).dropna()

        current_value = (
            current_feature_values[
                feature
            ]
        )

        if (
            reference_values.empty
            or pd.isna(
                current_value
            )
        ):
            current_feature_percentiles.append(
                np.nan
            )
        else:
            current_feature_percentiles.append(
                float(
                    (
                        reference_values
                        <= current_value
                    )
                    .mean()
                    * 100
                )
            )

    historical_movements = (
        _historical_period_movements(
            prediction_data=prediction_data,
            actual_column=actual_column,
            date_column=date_column,
            period_days=current_period_days,
            aggregation=aggregation,
        )
    )

    current_abs_movement = abs(
        actual_change
    )

    if historical_movements.empty:
        movement_percentile = np.nan
        movement_rank = 1
        movement_history_count = 0
    else:
        movement_percentile = float(
            (
                historical_movements
                <= current_abs_movement
            )
            .mean()
            * 100
        )

        movement_rank = int(
            (
                historical_movements
                > current_abs_movement
            )
            .sum()
            + 1
        )

        movement_history_count = int(
            len(
                historical_movements
            )
        )

    contributions = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "previous_value": (
                previous_feature_values.values
            ),
            "current_value": (
                current_feature_values.values
            ),
            "value_change": (
                current_feature_values.values
                - previous_feature_values.values
            ),
            "model_expected_contribution": (
                model_expected_contribution
            ),
            "proportional_actual_contribution": (
                proportional_actual_contribution
            ),
            "current_value_percentile": (
                current_feature_percentiles
            ),
        }
    )

    contributions[
        "absolute_model_expected_contribution"
    ] = (
        contributions[
            "model_expected_contribution"
        ]
        .abs()
    )

    contributions[
        "direction"
    ] = np.where(
        contributions[
            "model_expected_contribution"
        ] >= 0,
        "positive",
        "negative",
    )

    if predicted_change < 0:
        contributions = (
            contributions
            .sort_values(
                "model_expected_contribution",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )
    else:
        contributions = (
            contributions
            .sort_values(
                "model_expected_contribution",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    previous_period_label = _format_period(
        previous_start_date,
        previous_end_date,
    )

    current_period_label = _format_period(
        current_start_date,
        current_end_date,
    )

    summary = build_summary(
        model_stage=model_stage,
        previous_period=previous_period_label,
        current_period=current_period_label,
        previous_actual=previous_actual,
        current_actual=current_actual,
        actual_change=actual_change,
        previous_predicted=previous_predicted,
        current_predicted=current_predicted,
        predicted_change=predicted_change,
        actual_vs_expected_movement=(
            actual_vs_expected_movement
        ),
        contributions=contributions,
    )

    # This check catches an unexpected mismatch between the SHAP decomposition
    # and the aggregated model prediction movement.
    shap_reconstructed_change = float(
        (
            current_shap.sum()
            + expected_value
        )
        - (
            previous_shap.sum()
            + previous_expected_value
        )
    )

    if not np.isclose(
        shap_reconstructed_change,
        predicted_change,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError(
            "aggregated SHAP values do not reconstruct "
            "the modelled period movement. "
            f"SHAP: {shap_reconstructed_change:,.6f}; "
            f"prediction: {predicted_change:,.6f}"
        )

    return ExplanationResult(
        previous_period=previous_period_label,
        current_period=current_period_label,
        previous_actual=previous_actual,
        current_actual=current_actual,
        previous_predicted=previous_predicted,
        current_predicted=current_predicted,
        expected_value=expected_value,
        prediction_vs_baseline=prediction_vs_baseline,
        actual_change=actual_change,
        actual_change_pct=actual_change_pct,
        predicted_change=predicted_change,
        actual_vs_expected_movement=(
            actual_vs_expected_movement
        ),
        prediction_error=prediction_error,
        movement_scaling_factor=(
            movement_scaling_factor
        ),
        movement_percentile=movement_percentile,
        movement_rank=movement_rank,
        movement_history_count=(
            movement_history_count
        ),
        contribution_df=contributions,
        summary=summary,
    )