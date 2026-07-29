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
            "importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        chart_df["feature"],
        chart_df["importance"],
    )

    axis.set_title(
        f"Top {len(chart_df)} feature importances"
    )

    axis.set_xlabel(
        "Permutation importance"
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

def build_contribution_chart(
    explanation_result: ExplanationResult,
    top_n: int = 15,
) -> str:

    chart_df = (
        explanation_result
        .contributions
        .head(
            top_n
        )
        .sort_values(
            "contribution",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        chart_df["feature"],
        chart_df["contribution"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Drivers of predicted change"
    )

    axis.set_xlabel(
        "Contribution to predicted change"
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

    importance_image = (
        build_feature_importance_chart(
            importance_df=importance_df,
            top_n=20,
        )
    )

    contribution_image = (
        build_contribution_chart(
            explanation_result=explanation_result,
            top_n=15,
        )
    )

    summary_text = get_summary_text(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        model_stage=model_stage,
    )

    explanation_summary_df = pd.DataFrame(
        [
            {
                "previous_period": (
                    explanation_result
                    .previous_date
                    .date()
                ),
                "current_period": (
                    explanation_result
                    .current_date
                    .date()
                ),
                "previous_actual": (
                    explanation_result
                    .previous_actual
                ),
                "current_actual": (
                    explanation_result
                    .current_actual
                ),
                "actual_change": (
                    explanation_result
                    .actual_change
                ),
                "actual_change_pct": (
                    explanation_result
                    .actual_change_pct
                ),
                "predicted_change": (
                    explanation_result
                    .predicted_change
                ),
                "unexplained_change": (
                    explanation_result
                    .unexplained_change
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
        .sort_values(
            "date_day"
        )
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
                {summary_text}
            </p>
        </div>

        <div class="card">
            <h2>
                Explain latest change
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
                Drivers of predicted change
            </h2>

            <img
                class="chart"
                src="data:image/png;base64,{contribution_image}"
                alt="Drivers of predicted change"
            >

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        explanation_result
                        .contributions
                        .drop(
                            columns=[
                                "absolute_contribution",
                            ]
                        ),
                        decimals=3,
                        max_rows=20,
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

            <img
                class="chart"
                src="data:image/png;base64,{importance_image}"
                alt="Feature importance chart"
            >

            <div class="table-wrapper">
                {
                    dataframe_to_html(
                        importance_df,
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

