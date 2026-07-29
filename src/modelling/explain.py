"""
Explain changes between the latest two modelled periods.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class ExplanationResult:

    current_date: pd.Timestamp
    previous_date: pd.Timestamp

    current_actual: float
    previous_actual: float
    actual_change: float
    actual_change_pct: float

    current_predicted: float
    previous_predicted: float
    predicted_change: float

    unexplained_change: float

    contributions: pd.DataFrame
    summary: str


def build_summary(
    model_stage: str,
    model_grain: str,
    actual_change: float,
    predicted_change: float,
    unexplained_change: float,
    contributions: pd.DataFrame,
) -> str:

    stage_label = model_stage.upper()

    grain_labels = {
        "daily": "latest day",
        "weekly": "latest week",
        "monthly": "latest month",
    }

    period_label = grain_labels.get(
        model_grain,
        "latest period",
    )

    positive_drivers = (
        contributions
        .loc[
            contributions["contribution"] > 0
        ]
        .head(3)
    )

    negative_drivers = (
        contributions
        .loc[
            contributions["contribution"] < 0
        ]
        .sort_values(
            "contribution",
            ascending=True,
        )
        .head(3)
    )

    sentences = [
        (
            f"{stage_label} changed by "
            f"{actual_change:+,.1f} in the "
            f"{period_label}."
        ),
        (
            f"The model explained "
            f"{predicted_change:+,.1f} of this change."
        ),
    ]

    if not positive_drivers.empty:

        positive_text = ", ".join(
            (
                f"{row.feature} "
                f"({row.contribution:+,.1f})"
            )
            for row in positive_drivers.itertuples()
        )

        sentences.append(
            f"The largest positive drivers were "
            f"{positive_text}."
        )

    if not negative_drivers.empty:

        negative_text = ", ".join(
            (
                f"{row.feature} "
                f"({row.contribution:+,.1f})"
            )
            for row in negative_drivers.itertuples()
        )

        sentences.append(
            f"The largest negative drivers were "
            f"{negative_text}."
        )

    sentences.append(
        (
            f"The remaining unexplained difference was "
            f"{unexplained_change:+,.1f}."
        )
    )

    return " ".join(
        sentences
    )


def explain_latest_change(
    model,
    x_reference: pd.DataFrame,
    x_explain: pd.DataFrame,
    prediction_df: pd.DataFrame,
    model_stage: str,
    model_grain: str,
    date_column: str = "date_day",
    background_rows: int = 100,
    random_state: int = 42,
) -> ExplanationResult:

    if len(x_explain) < 2:

        raise ValueError(
            "at least two periods are required "
            "to explain change"
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
            (
                "prediction dataframe is missing: "
                f"{sorted(missing_columns)}"
            )
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
            (
                "prediction_df and x_explain must "
                "contain the same number of rows"
            )
        )

    prediction_data[
        date_column
    ] = pd.to_datetime(
        prediction_data[date_column],
        errors="coerce",
    )

    ordered_positions = (
        prediction_data[
            date_column
        ]
        .sort_values()
        .index
        .tolist()
    )

    previous_position = (
        ordered_positions[-2]
    )

    current_position = (
        ordered_positions[-1]
    )

    explain_positions = [
        previous_position,
        current_position,
    ]

    explain_features = (
        feature_data
        .iloc[
            explain_positions
        ]
        .copy()
    )

    reference_data = (
        x_reference
        .copy()
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

    previous_shap = np.asarray(
        shap_values.values[0]
    )

    current_shap = np.asarray(
        shap_values.values[1]
    )

    contribution_change = (
        current_shap
        - previous_shap
    )

    contributions = pd.DataFrame(
        {
            "feature": feature_data.columns,
            "previous_value": (
                explain_features
                .iloc[0]
                .values
            ),
            "current_value": (
                explain_features
                .iloc[1]
                .values
            ),
            "value_change": (
                explain_features
                .iloc[1]
                .values
                - explain_features
                .iloc[0]
                .values
            ),
            "contribution": contribution_change,
        }
    )

    contributions[
        "absolute_contribution"
    ] = contributions[
        "contribution"
    ].abs()

    contributions[
        "direction"
    ] = np.where(
        contributions[
            "contribution"
        ] >= 0,
        "positive",
        "negative",
    )

    contributions = (
        contributions
        .sort_values(
            "absolute_contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    previous_row = (
        prediction_data
        .iloc[
            previous_position
        ]
    )

    current_row = (
        prediction_data
        .iloc[
            current_position
        ]
    )

    previous_actual = float(
        previous_row[
            actual_column
        ]
    )

    current_actual = float(
        current_row[
            actual_column
        ]
    )

    previous_predicted = float(
        previous_row[
            predicted_column
        ]
    )

    current_predicted = float(
        current_row[
            predicted_column
        ]
    )

    actual_change = (
        current_actual
        - previous_actual
    )

    predicted_change = (
        current_predicted
        - previous_predicted
    )

    unexplained_change = (
        actual_change
        - predicted_change
    )

    if previous_actual == 0:

        actual_change_pct = np.nan

    else:

        actual_change_pct = (
            actual_change
            / previous_actual
            * 100
        )

    summary = build_summary(
        model_stage=model_stage,
        model_grain=model_grain,
        actual_change=actual_change,
        predicted_change=predicted_change,
        unexplained_change=unexplained_change,
        contributions=contributions,
    )

    return ExplanationResult(
        current_date=pd.Timestamp(
            current_row[
                date_column
            ]
        ),
        previous_date=pd.Timestamp(
            previous_row[
                date_column
            ]
        ),
        current_actual=current_actual,
        previous_actual=previous_actual,
        actual_change=actual_change,
        actual_change_pct=actual_change_pct,
        current_predicted=current_predicted,
        previous_predicted=previous_predicted,
        predicted_change=predicted_change,
        unexplained_change=unexplained_change,
        contributions=contributions,
        summary=summary,
    )