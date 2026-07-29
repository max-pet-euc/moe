"""
Build a self-contained HTML model report.
"""

from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modelling.explain import (
    ExplanationResult,
)


def ensure_directory(
    path,
) -> Path:

    directory = Path(
        path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def figure_to_base64(
    figure,
) -> str:

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=140,
    )

    buffer.seek(0)

    encoded_image = b64encode(
        buffer.read()
    ).decode(
        "utf-8"
    )

    buffer.close()
    plt.close(figure)

    return encoded_image


def build_actual_vs_predicted_chart(
    prediction_df: pd.DataFrame,
    model_stage: str,
) -> str:

    actual_column = (
        f"actual_{model_stage}"
    )

    predicted_column = (
        f"predicted_{model_stage}"
    )

    baseline_column = (
        f"baseline_{model_stage}"
    )

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        prediction_df["date_day"],
        prediction_df[actual_column],
        label="Actual",
        linewidth=2,
    )

    axis.plot(
        prediction_df["date_day"],
        prediction_df[predicted_column],
        label="Predicted",
        linewidth=2,
    )

    if baseline_column in prediction_df.columns:

        axis.plot(
            prediction_df["date_day"],
            prediction_df[baseline_column],
            label="Baseline",
            linestyle="--",
            linewidth=1.5,
        )

    axis.set_title(
        f"Actual vs predicted {model_stage.upper()}"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        model_stage.upper()
    )

    axis.legend()

    axis.grid(
        alpha=0.25
    )

    figure.autofmt_xdate()

    return figure_to_base64(
        figure
    )


def build_residual_chart(
    prediction_df: pd.DataFrame,
    model_stage: str,
) -> str:

    actual_column = (
        f"actual_{model_stage}"
    )

    predicted_column = (
        f"predicted_{model_stage}"
    )

    residuals = (
        prediction_df[actual_column]
        - prediction_df[predicted_column]
    )

    figure, axis = plt.subplots(
        figsize=(12, 4)
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.scatter(
        prediction_df["date_day"],
        residuals,
        alpha=0.75,
    )

    axis.set_title(
        "Prediction residuals"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Actual minus predicted"
    )

    axis.grid(
        alpha=0.25
    )

    figure.autofmt_xdate()

    return figure_to_base64(
        figure
    )


def build_feature_importance_chart(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> str:

    chart_df = (
        importance_df
        .head(top_n)
        .sort_values(
            "relative_importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        chart_df["feature"],
        chart_df["relative_importance"],
    )

    axis.set_title(
        f"Top {len(chart_df)} feature importances"
    )

    axis.set_xlabel(
        "Relative importance index (top feature = 100)"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    return figure_to_base64(
        figure
    )


def dataframe_to_html(
    dataframe: pd.DataFrame,
    decimals: int = 3,
    max_rows: int | None = None,
) -> str:

    output_df = dataframe.copy()

    if max_rows is not None:
        output_df = output_df.head(
            max_rows
        )

    numeric_columns = (
        output_df
        .select_dtypes(
            include=[
                np.number,
            ]
        )
        .columns
    )

    output_df[
        numeric_columns
    ] = output_df[
        numeric_columns
    ].round(
        decimals
    )

    return output_df.to_html(
        index=False,
        classes="report-table",
        border=0,
    )


def get_summary_text(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    model_stage: str,
) -> str:

    best_row = (
        comparison_df
        .loc[
            comparison_df["model"]
            == best_model_name
        ]
        .iloc[0]
    )

    baseline_rows = comparison_df.loc[
        comparison_df["model"]
        == "baseline_mean"
    ]

    best_rmse = float(
        best_row["rmse"]
    )

    best_mae = float(
        best_row["mae"]
    )

    best_r2 = float(
        best_row["r2"]
    )

    if not baseline_rows.empty:

        baseline_rmse = float(
            baseline_rows.iloc[0]["rmse"]
        )

        rmse_improvement = (
            (
                baseline_rmse
                - best_rmse
            )
            / baseline_rmse
            * 100
        )

        baseline_text = (
            f"The model improves RMSE by "
            f"{rmse_improvement:.1f}% "
            f"versus the mean baseline."
        )

    else:

        baseline_text = (
            "No baseline result was available."
        )

    return (
        f"The best-performing model for "
        f"{model_stage.upper()} was "
        f"{best_model_name}. "
        f"It achieved an RMSE of "
        f"{best_rmse:,.1f}, "
        f"an MAE of {best_mae:,.1f}, "
        f"and an R² of {best_r2:.3f}. "
        f"{baseline_text}"
    )

def build_driver_chart(
    contribution_df: pd.DataFrame,
    direction: str,
    top_n: int = 10,
) -> str:

    if direction == "positive":

        chart_df = (
            contribution_df
            .loc[
                contribution_df[
                    "model_expected_contribution"
                ] > 0
            ]
            .sort_values(
                "model_expected_contribution",
                ascending=False,
            )
            .head(top_n)
            .sort_values(
                "model_expected_contribution",
                ascending=True,
            )
        )

        title = "Drivers increasing expected movement"

    else:

        chart_df = (
            contribution_df
            .loc[
                contribution_df[
                    "model_expected_contribution"
                ] < 0
            ]
            .sort_values(
                "model_expected_contribution",
                ascending=True,
            )
            .head(top_n)
            .sort_values(
                "model_expected_contribution",
                ascending=False,
            )
        )

        title = "Drivers reducing expected movement"

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        chart_df["feature"],
        chart_df["model_expected_contribution"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Contribution to expected movement"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    return figure_to_base64(
        figure
    )


def build_movement_waterfall_chart(
    explanation_result: ExplanationResult,
    top_n: int = 8,
) -> str:

    contribution_df = (
        explanation_result
        .contribution_df
        .copy()
    )

    top_df = (
        contribution_df
        .sort_values(
            "absolute_model_expected_contribution",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    top_total = float(
        top_df[
            "model_expected_contribution"
        ].sum()
    )

    other_contribution = (
        explanation_result.predicted_change
        - top_total
    )

    labels = [
        "Previous prediction",
        *top_df["feature"].tolist(),
    ]

    movements = [
        explanation_result.previous_predicted,
        *top_df[
            "model_expected_contribution"
        ].tolist(),
    ]

    if abs(other_contribution) > 0.001:

        labels.append(
            "Other features"
        )

        movements.append(
            other_contribution
        )

    labels.append(
        "Current prediction"
    )

    starts = [0.0]
    heights = [
        explanation_result.previous_predicted
    ]

    running_total = (
        explanation_result.previous_predicted
    )

    for movement in movements[1:]:

        if movement >= 0:
            starts.append(
                running_total
            )
        else:
            starts.append(
                running_total + movement
            )

        heights.append(
            abs(movement)
        )

        running_total += movement

    starts.append(0.0)
    heights.append(
        explanation_result.current_predicted
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    x_positions = np.arange(
        len(labels)
    )

    axis.bar(
        x_positions,
        heights,
        bottom=starts,
    )

    axis.axhline(
        explanation_result.current_actual,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Current actual "
            f"({explanation_result.current_actual:,.1f})"
        ),
    )

    axis.set_title(
        "How the model moved from the previous to current prediction"
    )

    axis.set_ylabel(
        "Predicted value"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        labels,
        rotation=40,
        ha="right",
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    return figure_to_base64(
        figure
    )


def calculate_reliability_score(
    prediction_df: pd.DataFrame,
    model_stage: str,
) -> dict:

    actual_column = (
        f"actual_{model_stage}"
    )

    predicted_column = (
        f"predicted_{model_stage}"
    )

    reliability_df = (
        prediction_df[
            [
                actual_column,
                predicted_column,
            ]
        ]
        .dropna()
    )

    if reliability_df.empty:

        return {
            "score": np.nan,
            "label": "unavailable",
            "mae": np.nan,
            "typical_actual": np.nan,
        }

    absolute_errors = (
        reliability_df[actual_column]
        - reliability_df[predicted_column]
    ).abs()

    mae = float(
        absolute_errors.mean()
    )

    typical_actual = float(
        reliability_df[actual_column]
        .abs()
        .median()
    )

    if typical_actual == 0:
        score = np.nan
    else:
        score = float(
            np.clip(
                100
                * (
                    1
                    - mae
                    / typical_actual
                ),
                0,
                100,
            )
        )

    if np.isnan(score):
        label = "unavailable"
    elif score >= 80:
        label = "high"
    elif score >= 60:
        label = "moderate"
    else:
        label = "low"

    return {
        "score": score,
        "label": label,
        "mae": mae,
        "typical_actual": typical_actual,
    }


def build_surprise_text(
    explanation_result: ExplanationResult,
    model_stage: str,
) -> str:

    stage_label = model_stage.upper()

    movement_gap = (
        explanation_result.actual_vs_expected_movement
    )

    current_error = (
        explanation_result.prediction_error
    )

    if movement_gap > 0:
        movement_text = (
            f"Actual {stage_label} movement was "
            f"{abs(movement_gap):,.1f} better than expected."
        )
    elif movement_gap < 0:
        movement_text = (
            f"Actual {stage_label} movement was "
            f"{abs(movement_gap):,.1f} worse than expected."
        )
    else:
        movement_text = (
            f"Actual {stage_label} movement matched expectation."
        )

    if current_error > 0:
        level_text = (
            f"The current actual value finished "
            f"{abs(current_error):,.1f} above the prediction."
        )
    elif current_error < 0:
        level_text = (
            f"The current actual value finished "
            f"{abs(current_error):,.1f} below the prediction."
        )
    else:
        level_text = (
            "The current actual value matched the prediction."
        )

    investigation_text = (
        "This residual is not explained by SHAP. It may reflect "
        "operational changes, tracking effects, external demand, "
        "unmodelled variables, or normal prediction error."
    )

    return " ".join(
        [
            movement_text,
            level_text,
            investigation_text,
        ]
    )


def build_opportunity_df(
    importance_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    opportunity_df = (
        importance_df[
            [
                "feature",
                "relative_importance",
                "stability",
            ]
        ]
        .merge(
            contribution_df[
                [
                    "feature",
                    "previous_value",
                    "current_value",
                    "value_change",
                    "model_expected_contribution",
                ]
            ],
            on="feature",
            how="left",
        )
    )

    opportunity_df[
        "latest_direction"
    ] = np.select(
        [
            opportunity_df[
                "model_expected_contribution"
            ] > 0,
            opportunity_df[
                "model_expected_contribution"
            ] < 0,
        ],
        [
            "upward",
            "downward",
        ],
        default="neutral",
    )

    opportunity_df[
        "priority_score"
    ] = (
        opportunity_df[
            "relative_importance"
        ]
        * opportunity_df[
            "model_expected_contribution"
        ]
        .abs()
        .fillna(0)
    )

    return (
        opportunity_df
        .sort_values(
            [
                "priority_score",
                "relative_importance",
            ],
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


def build_executive_story(
    model_summary: str,
    explanation_result: ExplanationResult,
    reliability: dict,
    opportunity_df: pd.DataFrame,
    model_stage: str,
) -> str:

    stage_label = model_stage.upper()

    if explanation_result.predicted_change > 0:
        predicted_text = (
            f"The model expected {stage_label} to rise by "
            f"{abs(explanation_result.predicted_change):,.1f}."
        )
    elif explanation_result.predicted_change < 0:
        predicted_text = (
            f"The model expected {stage_label} to fall by "
            f"{abs(explanation_result.predicted_change):,.1f}."
        )
    else:
        predicted_text = (
            f"The model expected {stage_label} to remain unchanged."
        )

    if explanation_result.actual_change > 0:
        actual_text = (
            f"Actual {stage_label} rose by "
            f"{abs(explanation_result.actual_change):,.1f}."
        )
    elif explanation_result.actual_change < 0:
        actual_text = (
            f"Actual {stage_label} fell by "
            f"{abs(explanation_result.actual_change):,.1f}."
        )
    else:
        actual_text = (
            f"Actual {stage_label} was unchanged."
        )

    surprise_value = (
        explanation_result.actual_vs_expected_movement
    )

    if surprise_value > 0:
        surprise_text = (
            f"Performance was {abs(surprise_value):,.1f} "
            "better than the expected movement."
        )
    elif surprise_value < 0:
        surprise_text = (
            f"Performance was {abs(surprise_value):,.1f} "
            "worse than the expected movement."
        )
    else:
        surprise_text = (
            "Performance matched the expected movement."
        )

    if opportunity_df.empty:
        lever_text = (
            "No leading model lever was available."
        )
    else:
        leading_feature = str(
            opportunity_df.iloc[0]["feature"]
        )

        lever_text = (
            f"The leading model lever to review is "
            f"{leading_feature}."
        )

    if np.isnan(
        reliability["score"]
    ):
        reliability_text = (
            "Model reliability could not be calculated."
        )
    else:
        reliability_text = (
            f"The heuristic reliability score is "
            f"{reliability['score']:.0f}/100 "
            f"({reliability['label']})."
        )

    return " ".join(
        [
            predicted_text,
            actual_text,
            surprise_text,
            lever_text,
            reliability_text,
            model_summary,
        ]
    )


def build_model_report(
    model_stage: str,
    model_grain: str,
    target_column: str,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    model_df: pd.DataFrame,
    feature_columns: list[str],
    excluded_leakage_columns: list[str],
    train_rows: int,
    test_rows: int,
    test_start_date,
    output_paths: dict[str, Path],
    explanation_result: ExplanationResult,
    output_path,
) -> Path:

    output_path = Path(
        output_path
    )

    ensure_directory(
        output_path.parent
    )

    report_prediction_df = (
        prediction_df.copy()
    )

    report_prediction_df[
        "date_day"
    ] = pd.to_datetime(
        report_prediction_df["date_day"],
        errors="coerce",
    )

    actual_vs_predicted_image = (
        build_actual_vs_predicted_chart(
            prediction_df=report_prediction_df,
            model_stage=model_stage,
        )
    )

    residual_image = (
        build_residual_chart(
            prediction_df=report_prediction_df,
            model_stage=model_stage,
        )
    )

    report_importance_df = (
        importance_df.copy()
    )

    positive_importance = (
        report_importance_df["importance"]
        .clip(lower=0)
    )

    max_importance = float(
        positive_importance.max()
    )

    if max_importance > 0:
        report_importance_df[
            "relative_importance"
        ] = (
            positive_importance
            / max_importance
            * 100
        )
    else:
        report_importance_df[
            "relative_importance"
        ] = 0.0

    report_importance_df[
        "importance_variability"
    ] = (
        report_importance_df["importance_std"]
        / report_importance_df["importance"]
        .abs()
        .replace(0, np.nan)
    )

    report_importance_df[
        "stability"
    ] = np.select(
        [
            report_importance_df[
                "importance_variability"
            ] <= 0.15,
            report_importance_df[
                "importance_variability"
            ] <= 0.35,
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    report_importance_df = (
        report_importance_df
        .sort_values(
            "relative_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_image = (
        build_feature_importance_chart(
            importance_df=report_importance_df,
            top_n=20,
        )
    )

    positive_driver_image = (
        build_driver_chart(
            contribution_df=(
                explanation_result
                .contribution_df
            ),
            direction="positive",
            top_n=10,
        )
    )

    negative_driver_image = (
        build_driver_chart(
            contribution_df=(
                explanation_result
                .contribution_df
            ),
            direction="negative",
            top_n=10,
        )
    )

    waterfall_image = (
        build_movement_waterfall_chart(
            explanation_result=(
                explanation_result
            ),
            top_n=8,
        )
    )

    model_summary = get_summary_text(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        model_stage=model_stage,
    )

    reliability = calculate_reliability_score(
        prediction_df=report_prediction_df,
        model_stage=model_stage,
    )

    surprise_text = build_surprise_text(
        explanation_result=explanation_result,
        model_stage=model_stage,
    )

    opportunity_df = build_opportunity_df(
        importance_df=report_importance_df,
        contribution_df=(
            explanation_result
            .contribution_df
        ),
        top_n=10,
    )

    executive_story = build_executive_story(
        model_summary=model_summary,
        explanation_result=explanation_result,
        reliability=reliability,
        opportunity_df=opportunity_df,
        model_stage=model_stage,
    )

    explanation_summary_df = pd.DataFrame(
        [
            {
                "previous_period": explanation_result.previous_period,
                "current_period": explanation_result.current_period,
                "previous_actual": explanation_result.previous_actual,
                "current_actual": explanation_result.current_actual,
                "actual_change": explanation_result.actual_change,
                "previous_predicted": explanation_result.previous_predicted,
                "current_predicted": explanation_result.current_predicted,
                "predicted_change": explanation_result.predicted_change,
                "actual_vs_expected_movement": (
                    explanation_result.actual_vs_expected_movement
                ),
                "current_prediction_error": (
                    explanation_result.prediction_error
                ),
                "movement_scaling_factor": (
                    explanation_result.movement_scaling_factor
                ),
            }
        ]
    )

    reliability_df = pd.DataFrame(
        [
            {
                "reliability_score": reliability["score"],
                "reliability_label": reliability["label"],
                "historical_mae": reliability["mae"],
                "typical_actual_value": (
                    reliability["typical_actual"]
                ),
            }
        ]
    )

    model_config_df = pd.DataFrame(
        [
            {
                "model_stage": model_stage,
                "model_grain": model_grain,
                "target_column": target_column,
                "best_model": best_model_name,
                "data_rows": len(model_df),
                "feature_count": len(feature_columns),
                "excluded_leakage_columns": len(
                    excluded_leakage_columns
                ),
                "train_rows": train_rows,
                "test_rows": test_rows,
                "date_min": (
                    model_df["date_day"]
                    .min()
                    .date()
                ),
                "date_max": (
                    model_df["date_day"]
                    .max()
                    .date()
                ),
                "test_start_date": (
                    pd.Timestamp(
                        test_start_date
                    )
                    .date()
                ),
            }
        ]
    )

    recent_predictions_df = (
        report_prediction_df
        .sort_values(
            "date_day",
            ascending=False,
        )
        .head(20)
    )

    output_files_df = pd.DataFrame(
        [
            {
                "output": output_name,
                "path": str(output_file),
            }
            for output_name, output_file
            in output_paths.items()
        ]
    )

    positive_driver_df = (
        explanation_result
        .contribution_df
        .loc[
            explanation_result
            .contribution_df[
                "model_expected_contribution"
            ] > 0
        ]
        .sort_values(
            "model_expected_contribution",
            ascending=False,
        )
        .head(10)
    )

    negative_driver_df = (
        explanation_result
        .contribution_df
        .loc[
            explanation_result
            .contribution_df[
                "model_expected_contribution"
            ] < 0
        ]
        .sort_values(
            "model_expected_contribution",
            ascending=True,
        )
        .head(10)
    )

    confidence_score = reliability["score"]

    if np.isnan(confidence_score):
        confidence_width = 0
        confidence_display = "N/A"
    else:
        confidence_width = int(
            round(confidence_score)
        )
        confidence_display = (
            f"{confidence_score:.0f}/100"
        )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>
    {model_stage.upper()} {model_grain.title()} Model Report
    </title>

    <style>
        body {{
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
            margin: 0;
            background: #f4f6f8;
            color: #1f2933;
        }}

        .page {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px;
        }}

        .header {{
            background: #111827;
            color: white;
            padding: 28px 32px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            margin: 0 0 8px 0;
        }}

        .header p {{
            margin: 0;
            color: #d1d5db;
        }}

        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow:
                0 1px 3px rgba(0, 0, 0, 0.08);
        }}

        .card h2 {{
            margin-top: 0;
        }}

        .summary {{
            font-size: 18px;
            line-height: 1.6;
        }}

        .chart {{
            width: 100%;
            height: auto;
        }}

        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        .report-table th {{
            background: #eef2f7;
            text-align: left;
            padding: 10px;
            border-bottom: 2px solid #d9e0e8;
        }}

        .report-table td {{
            padding: 10px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .table-wrapper {{
            overflow-x: auto;
        }}

        .grid {{
            display: grid;
            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(450px, 1fr)
                );
            gap: 24px;
        }}

        .confidence-track {{
            width: 100%;
            background: #e5e7eb;
            border-radius: 999px;
            height: 22px;
            overflow: hidden;
        }}

        .confidence-fill {{
            width: {confidence_width}%;
            height: 100%;
            background: #111827;
        }}

        .metric {{
            font-size: 30px;
            font-weight: 700;
            margin: 8px 0;
        }}

        .note {{
            color: #667085;
            font-size: 14px;
            line-height: 1.5;
        }}

        .footer {{
            color: #667085;
            font-size: 13px;
            padding: 8px 0 24px 0;
        }}

        @media (
            max-width: 700px
        ) {{
            .page {{
                padding: 16px;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>

<body>
    <div class="page">

        <div class="header">
            <h1>
                {model_stage.upper()} {model_grain.title()} Model Report
            </h1>

            <p>
                Marketing Optimisation Engine
            </p>
        </div>

        <div class="card">
            <h2>
                Executive summary
            </h2>

            <p class="summary">
                {executive_story}
            </p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>
                    Latest movement
                </h2>

                <p class="summary">
                    {explanation_result.summary}
                </p>

                <div class="table-wrapper">
                    {
                        dataframe_to_html(
                            explanation_summary_df,
                            decimals=2,
                        )
                    }
                </div>
            </div>

            <div class="card">
                <h2>
                    Model reliability
                </h2>

                <div class="metric">
                    {confidence_display}
                </div>

                <div class="confidence-track">
                    <div class="confidence-fill"></div>
                </div>

                <p class="note">
                    This is a heuristic reliability score based on historical
                    mean absolute error relative to the typical actual value.
                    It is not a statistical confidence interval.
                </p>

                <div class="table-wrapper">
                    {
                        dataframe_to_html(
                            reliability_df,
                            decimals=2,
                        )
                    }
                </div>
            </div>
        </div>

        <div class="card">
            <h2>
                Prediction movement waterfall
            </h2>

            <p>
                The waterfall explains how the model moved from the previous
                prediction to the current prediction. The dashed line shows
                the current actual value for context.
            </p>

            <img
                class="chart"
                src="data:image/png;base64,{waterfall_image}"
                alt="Prediction movement waterfall"
            >
        </div>

        <div class="grid">
            <div class="card">
                <h2>
                    Drivers increasing expected movement
                </h2>

                <img
                    class="chart"
                    src="data:image/png;base64,{positive_driver_image}"
                    alt="Positive model drivers"
                >

                <div class="table-wrapper">
                    {
                        dataframe_to_html(
                            positive_driver_df[
                                [
                                    "feature",
                                    "previous_value",
                                    "current_value",
                                    "value_change",
                                    "model_expected_contribution",
                                    "proportional_actual_contribution",
                                ]
                            ],
                            decimals=3,
                        )
                    }
                </div>
            </div>

            <div class="card">
                <h2>
                    Drivers reducing expected movement
                </h2>

                <img
                    class="chart"
                    src="data:image/png;base64,{negative_driver_image}"
                    alt="Negative model drivers"
                >

                <div class="table-wrapper">
                    {
                        dataframe_to_html(
                            negative_driver_df[
                                [
                                    "feature",
                                    "previous_value",
                                    "current_value",
                                    "value_change",
                                    "model_expected_contribution",
                                    "proportional_actual_contribution",
                                ]
                            ],
                            decimals=3,
                        )
                    }
                </div>
            </div>
        </div>

        <div class="card">
            <h2>
                Biggest surprise
            </h2>

            <p class="summary">
                {surprise_text}
            </p>
        </div>

        <div class="card">
            <h2>
                Highest-priority model levers
            </h2>

            <p>
                These are the most important features to review, combining
                global relative importance with the size of their latest
                expected contribution. This is a prioritisation view, not a
                recommendation to increase or decrease spend and not a causal
                estimate.
            </p>

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        opportunity_df[
                            [
                                "feature",
                                "relative_importance",
                                "stability",
                                "previous_value",
                                "current_value",
                                "value_change",
                                "model_expected_contribution",
                                "latest_direction",
                                "priority_score",
                            ]
                        ],
                        decimals=3,
                    )
                }
            </div>
        </div>

        <div class="card">
            <h2>
                Model configuration
            </h2>

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        model_config_df
                    )
                }
            </div>
        </div>

        <div class="card">
            <h2>
                Model comparison
            </h2>

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        comparison_df
                    )
                }
            </div>
        </div>

        <div class="card">
            <h2>
                Actual vs predicted
            </h2>

            <img
                class="chart"
                src="data:image/png;base64,{actual_vs_predicted_image}"
                alt="Actual versus predicted chart"
            >
        </div>

        <div class="card">
            <h2>
                Prediction residuals
            </h2>

            <img
                class="chart"
                src="data:image/png;base64,{residual_image}"
                alt="Prediction residual chart"
            >
        </div>

        <div class="card">
            <h2>
                Feature importance
            </h2>

            <p>
                <strong>Relative importance</strong> sets the most important
                feature to 100 and expresses every other feature relative to
                it. A score of 50 means the feature is approximately half as
                influential as the leading feature in this model.
                <strong>Stability</strong> compares importance variation with
                the importance estimate itself: high means the result was
                consistent across shuffles, while low means it varied more.
                Raw importance and importance_std are retained for technical
                reference.
            </p>

            <img
                class="chart"
                src="data:image/png;base64,{importance_image}"
                alt="Feature importance chart"
            >

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        report_importance_df,
                        decimals=4,
                        max_rows=20,
                    )
                }
            </div>
        </div>

        <div class="card">
            <h2>
                Recent predictions
            </h2>

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        recent_predictions_df
                    )
                }
            </div>
        </div>

        <div class="card">
            <h2>
                Output files
            </h2>

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        output_files_df
                    )
                }
            </div>
        </div>

        <div class="footer">
            Generated by the Marketing Optimisation Engine.
        </div>

    </div>
</body>
</html>
"""

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    return output_path
