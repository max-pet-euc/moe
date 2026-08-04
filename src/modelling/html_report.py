"""
Build a self-contained HTML model report.

Report v2 is deliberately executive-first: scorecard, movement story,
waterfall, directional drivers, surprise, levers, then diagnostics.
"""

REPORT_VERSION = "2.1-20260729"
REPORT_FILENAME = "html_report_v2_20260729.py"

from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.modelling.explain import (
        ExplanationResult,
    )
except ImportError:
    from explain import (
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
            f"versus the mean baseline.\n"
        )

    else:

        baseline_text = (
            "No baseline result was available.\n"
        )

    return (
        f"The best-performing model for "
        f"{model_stage.upper()} was "
        f"{best_model_name}. "
        f"It achieved an RMSE of "
        f"{best_rmse:,.1f}, "
        f"an MAE of {best_mae:,.1f}, "
        f"and an R² of {best_r2:.3f}. "
        f"{baseline_text}\n"
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
            f"{abs(movement_gap):,.1f} better than expected.\n"
        )
    elif movement_gap < 0:
        movement_text = (
            f"Actual {stage_label} movement was "
            f"{abs(movement_gap):,.1f} worse than expected.\n"
        )
    else:
        movement_text = (
            f"Actual {stage_label} movement matched expectation.\n"
        )

    if current_error > 0:
        level_text = (
            f"The current actual value finished "
            f"{abs(current_error):,.1f} above the prediction.\n"
        )
    elif current_error < 0:
        level_text = (
            f"The current actual value finished "
            f"{abs(current_error):,.1f} below the prediction.\n"
        )
    else:
        level_text = (
            "The current actual value matched the prediction.\n"
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
            f"{abs(explanation_result.predicted_change):,.1f}.\n"
        )
    elif explanation_result.predicted_change < 0:
        predicted_text = (
            f"The model expected {stage_label} to fall by "
            f"{abs(explanation_result.predicted_change):,.1f}.\n"
        )
    else:
        predicted_text = (
            f"The model expected {stage_label} to remain unchanged.\n"
        )

    if explanation_result.actual_change > 0:
        actual_text = (
            f"Actual {stage_label} rose by "
            f"{abs(explanation_result.actual_change):,.1f}.\n"
        )
    elif explanation_result.actual_change < 0:
        actual_text = (
            f"Actual {stage_label} fell by "
            f"{abs(explanation_result.actual_change):,.1f}.\n"
        )
    else:
        actual_text = (
            f"Actual {stage_label} was unchanged.\n"
        )

    surprise_value = (
        explanation_result.actual_vs_expected_movement
    )

    if surprise_value > 0:
        surprise_text = (
            f"Performance was {abs(surprise_value):,.1f} "
            "better than the expected movement.\n"
        )
    elif surprise_value < 0:
        surprise_text = (
            f"Performance was {abs(surprise_value):,.1f} "
            "worse than the expected movement.\n"
        )
    else:
        surprise_text = (
            "Performance matched the expected movement.\n"
        )

    if opportunity_df.empty:
        lever_text = (
            "No leading model lever was available.\n"
        )
    else:
        leading_feature = str(
            opportunity_df.iloc[0]["feature"]
        )

        lever_text = (
            f"The leading model lever to review is "
            f"{leading_feature}.\n"
        )

    if np.isnan(
        reliability["score"]
    ):
        reliability_text = (
            "Model reliability could not be calculated.\n"
        )
    else:
        reliability_text = (
            f"The heuristic reliability score is "
            f"{reliability['score']:.0f}/100 "
            f"({reliability['label']}).\n"
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



def build_unusualness_text(
    explanation_result: ExplanationResult,
) -> str:

    percentile = explanation_result.movement_percentile
    rank = explanation_result.movement_rank
    history_count = explanation_result.movement_history_count

    if np.isnan(percentile) or history_count == 0:
        return "Historical movement context is unavailable."

    return (
        f"The latest absolute movement was larger than "
        f"{percentile:.0f}% of historical period-to-period movements "
        f"and ranks #{rank} out of {history_count}."
    )


def build_unusual_feature_df(
    contribution_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    output_df = contribution_df.copy()
    output_df["distance_from_median_percentile"] = (
        output_df["current_value_percentile"] - 50
    ).abs()

    return (
        output_df
        .sort_values(
            [
                "distance_from_median_percentile",
                "absolute_model_expected_contribution",
            ],
            ascending=False,
        )
        .head(top_n)
        [[
            "feature",
            "current_value",
            "current_value_percentile",
            "model_expected_contribution",
            "direction",
        ]]
        .reset_index(drop=True)
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

    unusualness_text = build_unusualness_text(
        explanation_result=explanation_result,
    )

    unusual_feature_df = build_unusual_feature_df(
        contribution_df=explanation_result.contribution_df,
        top_n=10,
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
                "movement_percentile": (
                    explanation_result.movement_percentile
                ),
                "movement_rank": explanation_result.movement_rank,
                "movement_history_count": (
                    explanation_result.movement_history_count
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

    def signed(value: float) -> str:
        return f"{value:+,.1f}"

    def movement_label(value: float) -> str:
        if value > 0:
            return "increase"
        if value < 0:
            return "decrease"
        return "no change"

    top_positive = (
        positive_driver_df.iloc[0]["feature"]
        if not positive_driver_df.empty
        else "No material upward driver"
    )

    top_negative = (
        negative_driver_df.iloc[0]["feature"]
        if not negative_driver_df.empty
        else "No material downward driver"
    )

    surprise_class = (
        "good"
        if explanation_result.actual_vs_expected_movement > 0
        else "bad"
        if explanation_result.actual_vs_expected_movement < 0
        else "neutral"
    )

    if np.isnan(
        explanation_result.actual_change_pct
    ):
        actual_change_pct_display = "N/A"
    else:
        actual_change_pct_display = (
            f"{explanation_result.actual_change_pct:+,.1f}%"
        )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_stage.upper()} Predictive Model: Report v{REPORT_VERSION}</title>
    <style>
        :root {{
            --ink: #172033;
            --muted: #667085;
            --line: #e6eaf0;
            --surface: #ffffff;
            --page: #f7f8fb;
            --brand: #5b4bdb;
            --brand-soft: #eeecff;
            --good: #0b7a53;
            --good-soft: #e8f7f0;
            --bad: #b42318;
            --bad-soft: #fff0ee;
            --warning: #9a6700;
            --warning-soft: #fff7df;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            background: var(--page);
            color: var(--ink);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .page {{ max-width: 1380px; margin: 0 auto; padding: 28px; }}

        .hero {{
            padding: 34px;
            border-radius: 20px;
            color: white;
            background: linear-gradient(135deg, #171b36 0%, #4438a8 100%);
            box-shadow: 0 18px 50px rgba(35, 28, 96, 0.20);
            margin-bottom: 22px;
        }}

        .eyebrow {{
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: 12px;
            font-weight: 700;
            opacity: .78;
        }}
        
        .period-label {{
            margin-top: 10px;
            font-size: 15px;
            color: #d9d5ff;
            font-weight: 600;
        }}

        .hero h1 {{ margin: 8px 0 8px; font-size: 38px; }}
        .hero p {{ max-width: 980px; margin: 0; line-height: 1.65; font-size: 18px; color: #e4e1ff; }}

        .scorecard {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }}

        .metric-card, .card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(16, 24, 40, 0.05);
        }}

        .metric-card {{ padding: 18px; }}
        .metric-label {{ color: var(--muted); font-size: 13px; font-weight: 650; }}
        .metric-value {{ font-size: 30px; font-weight: 760; margin: 7px 0 3px; }}
        .metric-sub {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
        .metric-card.good {{ background: var(--good-soft); border-color: #b7e5d1; }}
        .metric-card.bad {{ background: var(--bad-soft); border-color: #ffc9c3; }}
        .metric-card.neutral {{ background: var(--brand-soft); border-color: #d7d1ff; }}

        .card {{ padding: 24px; margin-bottom: 22px; }}
        .card h2 {{ margin: 0 0 8px; font-size: 22px; }}
        .card h3 {{ margin: 0 0 6px; font-size: 16px; }}
        .section-intro {{ color: var(--muted); margin: 0 0 18px; line-height: 1.55; }}
        .story {{ font-size: 18px; line-height: 1.65; margin: 0; }}

        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}

        .driver-callout {{
            border-radius: 14px;
            padding: 16px;
            border: 1px solid var(--line);
            background: #fbfcfe;
        }}
        .driver-callout.up {{ border-left: 5px solid var(--good); }}
        .driver-callout.down {{ border-left: 5px solid var(--bad); }}
        .driver-name {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; overflow-wrap: anywhere; }}

        .chart {{ width: 100%; height: auto; display: block; }}
        .table-wrapper {{ overflow-x: auto; }}
        .report-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .report-table th {{ text-align: left; background: #f2f4f7; padding: 10px; border-bottom: 2px solid #d8dee8; white-space: nowrap; }}
        .report-table td {{ padding: 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
        .report-table tr:hover td {{ background: #fafaff; }}

        .confidence-track {{ height: 12px; background: #e6e8ef; border-radius: 99px; overflow: hidden; margin: 12px 0 8px; }}
        .confidence-fill {{ width: {confidence_width}%; height: 100%; background: linear-gradient(90deg, #7c6df2, #3f32a8); }}
        .pill {{ display: inline-block; padding: 5px 9px; border-radius: 999px; background: #f0f2f5; font-size: 12px; font-weight: 650; }}
        .note {{ color: var(--muted); font-size: 13px; line-height: 1.55; }}

        details {{ border-top: 1px solid var(--line); padding-top: 16px; margin-top: 16px; }}
        summary {{ cursor: pointer; font-weight: 700; color: #3f32a8; }}
        .footer {{ color: var(--muted); font-size: 12px; padding: 4px 2px 24px; }}

        @media (max-width: 980px) {{
            .scorecard {{ grid-template-columns: repeat(2, 1fr); }}
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 600px) {{
            .page {{ padding: 14px; }}
            .hero {{ padding: 24px; }}
            .hero h1 {{ font-size: 29px; }}
            .scorecard {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
<div class="page">
    <section class="hero">
        <div class="eyebrow">Report v{REPORT_VERSION}</div>
        <h1>Marketing Optimisation Engine - {model_stage.upper()} Model</h1>
        <p class="period-label">
            {explanation_result.current_period}
            compared with
            {explanation_result.previous_period}
        </p>
        <p>{executive_story} {unusualness_text}</p>
    </section>

    <section class="scorecard">
        <div class="metric-card">
            <div class="metric-label">Current period actual</div>
            <div class="metric-value">{explanation_result.current_actual:,.1f}</div>
            <div class="metric-sub">
                {explanation_result.current_period}
            </div>
        </div>
        <div class="metric-card neutral">
            <div class="metric-label">Actual movement</div>
            <div class="metric-value">{signed(explanation_result.actual_change)}</div>
            <div class="metric-sub">Observed {movement_label(explanation_result.actual_change)}</div>
        </div>
        <div class="metric-card neutral">
            <div class="metric-label">Expected movement</div>
            <div class="metric-value">{signed(explanation_result.predicted_change)}</div>
            <div class="metric-sub">Model prediction vs previous period</div>
        </div>
        <div class="metric-card {surprise_class}">
            <div class="metric-label">Movement surprise</div>
            <div class="metric-value">{signed(explanation_result.actual_vs_expected_movement)}</div>
            <div class="metric-sub">Actual minus expected movement</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Previous period actual</div>
            <div class="metric-value">{explanation_result.previous_actual:,.1f}</div>
            <div class="metric-sub">
                {explanation_result.previous_period}
            </div>
        </div>
        <div class="metric-sub">
            {actual_change_pct_display}
            versus previous period
        </div>
    </section>

    <section class="card">
        <h2>What changed?</h2>
        <p class="story">{explanation_result.summary}</p>
    </section>

    <section class="card">
        <h2>Why the model changed</h2>
        <p class="section-intro">
            The waterfall starts with the model's total prediction for
            {explanation_result.previous_period}, adds the summed daily SHAP
            contribution changes, and ends with the model's total prediction for
            {explanation_result.current_period}.
        </p>
        <img class="chart" src="data:image/png;base64,{waterfall_image}" alt="Prediction movement waterfall">
    </section>

    <section class="grid-2">
        <div class="card">
            <h2>Upward pressure</h2>
            <div class="driver-callout up">
                <h3>Largest upward driver</h3>
                <div class="driver-name">{top_positive}</div>
            </div>
            <img class="chart" src="data:image/png;base64,{positive_driver_image}" alt="Positive model drivers">
            <div class="table-wrapper">{dataframe_to_html(positive_driver_df[["feature", "previous_value", "current_value", "value_change", "model_expected_contribution"]], decimals=3)}</div>
        </div>
        <div class="card">
            <h2>Downward pressure</h2>
            <div class="driver-callout down">
                <h3>Largest downward driver</h3>
                <div class="driver-name">{top_negative}</div>
            </div>
            <img class="chart" src="data:image/png;base64,{negative_driver_image}" alt="Negative model drivers">
            <div class="table-wrapper">{dataframe_to_html(negative_driver_df[["feature", "previous_value", "current_value", "value_change", "model_expected_contribution"]], decimals=3)}</div>
        </div>
    </section>

    <section class="grid-2">
        <div class="card">
            <h2>How much should we trust this?</h2>
            <div class="metric-value">{confidence_display}</div>
            <div class="confidence-track"><div class="confidence-fill"></div></div>
            <p class="note">This is a practical reliability indicator based on historical MAE relative to the typical actual value. It is not a probability and not a statistical confidence interval.</p>
            <div class="table-wrapper">{dataframe_to_html(reliability_df, decimals=2)}</div>
        </div>
    </section>

    <section class="card">
        <h2>How unusual is this movement?</h2>
        <p class="story">{unusualness_text}</p>
        <p class="note">This ranks the absolute actual movement against all historical movements available in the prediction dataset.</p>
    </section>

    <section class="card">
        <h2>Unusually high or low feature values</h2>
        <p class="section-intro">Average daily feature values during the selected period compared with the historical reference distribution.</p>
        <div class="table-wrapper">{dataframe_to_html(unusual_feature_df, decimals=2, max_rows=10)}</div>
    </section>

    <section class="card">
        <h2>Priority levers to review</h2>
        <p class="section-intro">This ranking combines global feature importance with how active each feature was in the selected predicted movement. It identifies where to investigate first; it does not prove causality or prescribe spend changes.</p>
        <div class="table-wrapper">{dataframe_to_html(opportunity_df[["feature", "relative_importance", "stability", "previous_value", "current_value", "value_change", "model_expected_contribution", "latest_direction", "priority_score"]], decimals=3)}</div>
    </section>

    <section class="card">
        <h2>Performance trend</h2>
        <p class="section-intro">Actual and predicted values across the evaluation period.</p>
        <img class="chart" src="data:image/png;base64,{actual_vs_predicted_image}" alt="Actual versus predicted chart">
    </section>

    <section class="card">
        <h2>Model diagnostics</h2>
        <p class="section-intro">Technical detail is retained below, but deliberately placed after the business story.</p>
        <div class="grid-2">
            <div>
                <h3>Residuals</h3>
                <img class="chart" src="data:image/png;base64,{residual_image}" alt="Prediction residual chart">
            </div>
            <div>
                <h3>Feature importance</h3>
                <img class="chart" src="data:image/png;base64,{importance_image}" alt="Feature importance chart">
            </div>
        </div>
        <details>
            <summary>Open technical tables</summary>
            <h3>Model configuration</h3>
            <div class="table-wrapper">{dataframe_to_html(model_config_df)}</div>
            <h3>Model comparison</h3>
            <div class="table-wrapper">{dataframe_to_html(comparison_df)}</div>
            <h3>Feature importance detail</h3>
            <div class="table-wrapper">{dataframe_to_html(report_importance_df, decimals=4, max_rows=20)}</div>
            <h3>Recent predictions</h3>
            <div class="table-wrapper">{dataframe_to_html(recent_predictions_df)}</div>
            <h3>Output files</h3>
            <div class="table-wrapper">{dataframe_to_html(output_files_df)}</div>
        </details>
    </section>

    <div class="footer">Generated by the Marketing Optimisation Engine · report schema v{REPORT_VERSION}</div>
</div>
</body>
</html>
"""

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    return output_path