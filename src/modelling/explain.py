"""
Explain movement between the latest two modelled periods.
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

    contribution_df: pd.DataFrame
    summary: str

    @property
    def unexplained_change(self) -> float:
        """Backward-compatible alias."""

        return self.actual_vs_expected_movement


def build_summary(
    model_stage: str,
    model_grain: str,
    previous_actual: float,
    current_actual: float,
    actual_change: float,
    previous_predicted: float,
    current_predicted: float,
    predicted_change: float,
    actual_vs_expected_movement: float,
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

    if actual_change < 0:
        actual_direction = "fell"
    elif actual_change > 0:
        actual_direction = "rose"
    else:
        actual_direction = "was unchanged"

    if predicted_change < 0:
        predicted_direction = "a fall"
        contributor_direction = "lowering"

        primary_contributors = (
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

    elif predicted_change > 0:
        predicted_direction = "an increase"
        contributor_direction = "increasing"

        primary_contributors = (
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

    else:
        predicted_direction = "no change"
        contributor_direction = "affecting"
        primary_contributors = contributions.head(0)

    if actual_change == 0:
        actual_sentence = (
            f"Actual {stage_label} was unchanged at "
            f"{current_actual:,.1f} in the {period_label}."
        )
    else:
        actual_sentence = (
            f"Actual {stage_label} {actual_direction} by "
            f"{abs(actual_change):,.1f}, from "
            f"{previous_actual:,.1f} to "
            f"{current_actual:,.1f}, in the {period_label}."
        )

    if predicted_change == 0:
        prediction_sentence = (
            "The model expected no change, with the prediction "
            f"remaining at {current_predicted:,.1f}."
        )
    else:
        prediction_sentence = (
            f"The model expected {predicted_direction} of "
            f"{abs(predicted_change):,.1f}, from "
            f"{previous_predicted:,.1f} to "
            f"{current_predicted:,.1f}."
        )

    sentences = [
        actual_sentence,
        prediction_sentence,
    ]

    if not primary_contributors.empty:

        contributor_text = ", ".join(
            (
                f"{row.feature} "
                f"({row.model_expected_contribution:+,.1f} model, "
                f"{row.proportional_actual_contribution:+,.1f} proportional actual)"
            )
            for row in primary_contributors.itertuples()
        )

        sentences.append(
            (
                f"The strongest contributors {contributor_direction} "
                f"the model's expected movement were {contributor_text}."
            )
        )

    if actual_vs_expected_movement > 0:
        gap_text = "outperformed"
    elif actual_vs_expected_movement < 0:
        gap_text = "underperformed"
    else:
        gap_text = "matched"

    if actual_vs_expected_movement == 0:
        sentences.append(
            "The actual movement matched the model's expectation."
        )
    else:
        sentences.append(
            (
                f"Actual performance {gap_text} the model's "
                f"expected movement by "
                f"{abs(actual_vs_expected_movement):,.1f}."
            )
        )

    sentences.append(
        (
            "Proportional actual contributions scale the model's "
            "SHAP movement attribution to the observed actual movement; "
            "they are directional estimates rather than causal measurements."
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
            "to explain the latest movement"
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
        prediction_data[
            date_column
        ],
        errors="coerce",
    )

    if prediction_data[
        date_column
    ].isna().any():

        raise ValueError(
            (
                f"{date_column} contains values "
                "that could not be converted to dates"
            )
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
    ).reshape(-1)

    current_shap = np.asarray(
        shap_values.values[1]
    ).reshape(-1)

    model_expected_contribution = (
        current_shap
        - previous_shap
    )

    current_base_value = np.asarray(
        shap_values.base_values[1]
    ).reshape(-1)

    expected_value = float(
        current_base_value[0]
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

    if previous_actual == 0:

        actual_change_pct = np.nan

    else:

        actual_change_pct = (
            actual_change
            / previous_actual
            * 100
        )

    if predicted_change == 0:

        movement_scaling_factor = np.nan

    else:

        movement_scaling_factor = (
            actual_change
            / predicted_change
        )

    proportional_actual_contribution = (
        model_expected_contribution
        * movement_scaling_factor
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
            "model_expected_contribution": (
                model_expected_contribution
            ),
            "proportional_actual_contribution": (
                proportional_actual_contribution
            ),
        }
    )

    contributions[
        "absolute_model_expected_contribution"
    ] = contributions[
        "model_expected_contribution"
    ].abs()

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

    summary = build_summary(
        model_stage=model_stage,
        model_grain=model_grain,
        previous_actual=previous_actual,
        current_actual=current_actual,
        actual_change=actual_change,
        previous_predicted=previous_predicted,
        current_predicted=current_predicted,
        predicted_change=predicted_change,
        actual_vs_expected_movement=actual_vs_expected_movement,
        contributions=contributions,
    )

    return ExplanationResult(
        previous_period=pd.Timestamp(
            previous_row[
                date_column
            ]
        ),
        current_period=pd.Timestamp(
            current_row[
                date_column
            ]
        ),
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
        contribution_df=contributions,
        summary=summary,
    )