
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config.settings import (
    ALWAYS_EXCLUDED_COLUMNS,
    CUSTOM_EXCLUDE_COLUMNS,
    ENGINEERED_DIR,
    FUNNEL_ORDER,
    MODEL_OUTPUT_DIR,
    MODEL_STAGE,
    MODEL_TARGETS,
    PREDICTION_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    STAGE_ALIASES,
    TEST_DAYS,
    MODEL_GRAIN,
    MODEL_GRAIN_COLUMNS,
    VALID_MODEL_GRAINS,
    REPORT_START_DATE,
    REPORT_END_DATE,
    REPORT_PERIOD_AGGREGATION,
    VALID_REPORT_PERIOD_AGGREGATIONS,
)

from src.modelling.feature_selection import select_features
from src.modelling.model_registry import get_models
from src.modelling.trainer import fit_models
from src.modelling.evaluation import evaluate_models
from src.modelling.reporting import (
    save_dataframe,
    save_model,
)
from src.modelling.html_report import (
    build_model_report,
)
from src.modelling.explain import (
    explain_period_change,
)

#=========================================================
#==config

model_stage = MODEL_STAGE
test_days = TEST_DAYS

funnel_order = FUNNEL_ORDER
model_targets = MODEL_TARGETS
stage_aliases = STAGE_ALIASES
excluded_columns = (
    ALWAYS_EXCLUDED_COLUMNS
    | CUSTOM_EXCLUDE_COLUMNS
)

model_output_dir = MODEL_OUTPUT_DIR
prediction_output_dir = (
    PREDICTION_OUTPUT_DIR
)

input_file = (
    ENGINEERED_DIR
    / "data_features.csv"
)

if MODEL_GRAIN not in VALID_MODEL_GRAINS:
    raise ValueError(
        f"invalid model grain: {MODEL_GRAIN}"
    )

model_grain = MODEL_GRAIN

model_date_column = (
    MODEL_GRAIN_COLUMNS[
        model_grain
    ]
)

#=========================================================
#==config dates

report_start_date = pd.Timestamp(
    REPORT_START_DATE
).normalize()

report_end_date = pd.Timestamp(
    REPORT_END_DATE
).normalize()

if report_start_date > report_end_date:
    raise ValueError(
        "REPORT_START_DATE must be on or before "
        "REPORT_END_DATE"
    )

if (
    REPORT_PERIOD_AGGREGATION
    not in VALID_REPORT_PERIOD_AGGREGATIONS
):
    raise ValueError(
        "invalid REPORT_PERIOD_AGGREGATION: "
        f"{REPORT_PERIOD_AGGREGATION}"
    )

report_period_days = (
    report_end_date
    - report_start_date
).days + 1

previous_end_date = (
    report_start_date
    - pd.Timedelta(days=1)
)

previous_start_date = (
    previous_end_date
    - pd.Timedelta(
        days=report_period_days - 1
    )
)


#=========================================================
#==helpers

def clean_numeric_series(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "£",
            "",
            regex=False,
        )
        .str.replace(
            "%",
            "",
            regex=False,
        )
        .str.strip(),
        errors="coerce",
    )


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:

    return {
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "r2": r2_score(
            actual,
            predicted,
        ),
    }


def get_leakage_stages(
    target_stage: str,
) -> list[str]:

    if target_stage not in funnel_order:
        raise ValueError(
            f"unknown model stage: {target_stage}"
        )

    target_index = funnel_order.index(
        target_stage
    )

    return funnel_order[target_index:]


def contains_stage_marker(
    column: str,
    stages: list[str],
) -> bool:

    column_lower = column.lower()

    markers = [
        alias
        for stage in stages
        for alias in stage_aliases[stage]
    ]

    return any(
        re.search(
            rf"(?:^|_){re.escape(marker)}(?:_|$)",
            column_lower,
        )
        is not None
        for marker in markers
    )


#=========================================================
#==resolve model config

if model_stage not in model_targets:
    raise ValueError(
        f"model_stage must be one of: "
        f"{list(model_targets)}"
    )

target_column = model_targets[
    model_stage
]

leakage_stages = get_leakage_stages(
    model_stage
)


#=========================================================
#==load data

df = pd.read_csv(
    input_file,
    low_memory=False,
)

df["date_day"] = pd.to_datetime(
    df["date_day"],
    errors="coerce",
)

if target_column not in df.columns:
    raise ValueError(
        f"target column not found: {target_column}"
    )

df[target_column] = clean_numeric_series(
    df[target_column]
)


#=========================================================
#==filter model data

model_df = (
    df
    .loc[
        df["date_day"].notna()
        & df[target_column].notna()
        & (
            df["date_day"]
            <= report_end_date
        )
    ]
    .sort_values(
        "date_day"
    )
    .reset_index(
        drop=True
    )
)

if model_df.empty:
    raise ValueError(
        "no rows available for modelling"
    )

available_dates = set(
    model_df["date_day"]
    .dropna()
    .dt.normalize()
)

expected_current_dates = set(
    pd.date_range(
        report_start_date,
        report_end_date,
        freq="D",
    )
)

expected_previous_dates = set(
    pd.date_range(
        previous_start_date,
        previous_end_date,
        freq="D",
    )
)

current_dates = sorted(
    available_dates.intersection(
        expected_current_dates
    )
)

previous_dates = sorted(
    available_dates.intersection(
        expected_previous_dates
    )
)

missing_current_dates = sorted(
    expected_current_dates
    - available_dates
)

missing_previous_dates = sorted(
    expected_previous_dates
    - available_dates
)

if missing_current_dates:
    print(
        "Warning: missing current reporting dates: "
        + ", ".join(
            str(date.date())
            for date in missing_current_dates
        )
    )

if missing_previous_dates:
    print(
        "Warning: missing previous reporting dates: "
        + ", ".join(
            str(date.date())
            for date in missing_previous_dates
        )
    )

if not current_dates:
    raise ValueError(
        "no available dates found in the "
        "current reporting period"
    )

if not previous_dates:
    raise ValueError(
        "no available dates found in the "
        "previous reporting period"
    )

# Keep both periods comparable by using the
# same number of available days.
comparison_days = min(
    len(current_dates),
    len(previous_dates),
)

current_dates = current_dates[
    -comparison_days:
]

previous_dates = previous_dates[
    -comparison_days:
]

report_start_date = min(current_dates)
report_end_date = max(current_dates)

previous_start_date = min(previous_dates)
previous_end_date = max(previous_dates)

print(
    "\nEffective current reporting period:",
    report_start_date.date(),
    "to",
    report_end_date.date(),
    f"({len(current_dates)} available days)",
)

print(
    "Effective previous reporting period:",
    previous_start_date.date(),
    "to",
    previous_end_date.date(),
    f"({len(previous_dates)} available days)",
)

#=========================================================
#==select features

selection = select_features(
    model_df=model_df,
    model_stage=model_stage,
)

x = selection.x
y = selection.y

feature_columns = (
    selection.feature_columns
)

excluded_leakage_columns = (
    selection.leakage_columns
)


#=========================================================
#==time split

max_date = model_df["date_day"].max()

test_start_date = (
    max_date
    - pd.Timedelta(
        days=test_days - 1
    )
)

if previous_start_date < test_start_date:
    raise ValueError(
        "selected comparison periods fall outside "
        "the model test window. Increase TEST_DAYS."
    )

train_mask = (
    model_df["date_day"]
    < test_start_date
)

test_mask = (
    model_df["date_day"]
    >= test_start_date
)

x_train = x.loc[train_mask]
x_test = x.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

date_test = model_df.loc[
    test_mask,
    "date_day",
]

if x_train.empty:
    raise ValueError(
        "training dataset is empty"
    )

if x_test.empty:
    raise ValueError(
        "test dataset is empty"
    )


#=========================================================
#==baseline

baseline_prediction = np.repeat(
    y_train.mean(),
    len(y_test),
)

baseline_metrics = calculate_metrics(
    actual=y_test,
    predicted=baseline_prediction,
)

#=========================================================
#==train models

models = get_models()

trained_models = fit_models(
    models=models,
    x_train=x_train,
    y_train=y_train,
)

#=========================================================
#==evaluate models

evaluation_result = evaluate_models(
    trained_models=trained_models,
    x_test=x_test,
    y_test=y_test,
)

metrics_df = (
    evaluation_result.metrics
)

model_predictions = (
    evaluation_result.predictions
)

#=========================================================
#==model comparison

baseline_row = pd.DataFrame(
    [
        {
            "model": "baseline_mean",
            "mae": baseline_metrics["mae"],
            "rmse": baseline_metrics["rmse"],
            "r2": baseline_metrics["r2"],
            "training_seconds": 0.0,
        }
    ]
)

comparison_df = (
    pd.concat(
        [
            baseline_row,
            metrics_df,
        ],
        ignore_index=True,
    )
    .sort_values(
        "rmse"
    )
    .reset_index(
        drop=True
    )
)

best_model_name = metrics_df.loc[
    0,
    "model",
]

best_model = trained_models[
    best_model_name
]

best_prediction = model_predictions[
    best_model_name
]


#=========================================================
#==reporting on predictions

prediction_df = pd.DataFrame(
    {
        "date_day": date_test.values,
        f"actual_{model_stage}": y_test.values,
        f"baseline_{model_stage}": baseline_prediction,
        f"predicted_{model_stage}": best_prediction,
        "model": best_model_name,
    }
)

prediction_file = (
    prediction_output_dir
    / f"{model_stage}_{model_grain}_predictions.csv"
)

save_dataframe(
    dataframe=prediction_df,
    output_path=prediction_file,
)


#=========================================================
#==permutation importance

importance_result = permutation_importance(
    best_model,
    x_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

importance_df = (
    pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                importance_result
                .importances_mean
            ),
            "importance_std": (
                importance_result
                .importances_std
            ),
        }
    )
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

importance_file = (
    model_output_dir
    / f"{model_stage}_{model_grain}_feature_importance.csv"
)

save_dataframe(
    dataframe=importance_df,
    output_path=importance_file,
)


#=========================================================
#==save feature list

feature_list_df = pd.DataFrame(
    {
        "feature": feature_columns,
    }
)

feature_list_file = (
    model_output_dir
    / f"{model_stage}_{model_grain}_features.csv"
)

save_dataframe(
    dataframe=feature_list_df,
    output_path=feature_list_file,
)


#=========================================================
#==save metrics

metrics_file = (
    model_output_dir
    / f"{model_stage}_{model_grain}_model_comparison.csv"
)

save_dataframe(
    dataframe=comparison_df,
    output_path=metrics_file,
)


#=========================================================
#==save best model

model_file = (
    model_output_dir
    / (
        f"{model_stage}_"
        f"{model_grain}_"
        f"{best_model_name}.joblib"
    )
)

save_model(
    model=best_model,
    output_path=model_file,
)


#=========================================================
#==output

print("\n=====================================")
print("==model config")
print(f"model stage: {model_stage}")
print(f"target: {target_column}")
print(
    "excluded leakage stages: "
    + ", ".join(leakage_stages)
)
print(
    "excluded leakage columns: "
    f"{len(excluded_leakage_columns):,}"
)

print("\n=====================================")
print("==model dataset")
print(f"rows: {len(model_df):,}")
print(f"features: {len(feature_columns):,}")
print(
    f"date min: "
    f"{model_df['date_day'].min().date()}"
)
print(
    f"date max: "
    f"{model_df['date_day'].max().date()}"
)
print(f"train rows: {len(x_train):,}")
print(f"test rows: {len(x_test):,}")
print(
    f"test start: "
    f"{test_start_date.date()}"
)


print("\n=====================================")
print("==model comparison")
print(
    comparison_df
    .round(3)
    .to_string(
        index=False
    )
)


print("\n=====================================")
print("==best model")
print(best_model_name)


print("\n=====================================")
print("==top features")
print(
    importance_df
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


print("\n=====================================")
print("==outputs")
print(f"model: {model_file}")
print(f"metrics: {metrics_file}")
print(f"predictions: {prediction_file}")
print(
    f"feature importance: "
    f"{importance_file}"
)
print(f"feature list: {feature_list_file}")


print("\n=====================================")
print("==reporting period")
print(
    "current: "
    f"{report_start_date.date()} "
    f"to {report_end_date.date()}"
)
print(
    "previous: "
    f"{previous_start_date.date()} "
    f"to {previous_end_date.date()}"
)
print(
    "aggregation: "
    f"{REPORT_PERIOD_AGGREGATION}"
)

#=========================================================
#==html output

report_file = (
    REPORT_OUTPUT_DIR
    / (
        f"{model_stage}_"
        f"{model_grain}_"
        f"model_report.html"
    )
)

#=========================================================
#==explain latest change

explanation_result = (
    explain_period_change(
        model=best_model,
        x_reference=x_train,
        x_explain=x_test,
        prediction_df=prediction_df,
        model_stage=model_stage,
        model_grain=model_grain,
        current_start_date=report_start_date,
        current_end_date=report_end_date,
        previous_start_date=previous_start_date,
        previous_end_date=previous_end_date,
        aggregation=(
            REPORT_PERIOD_AGGREGATION
        ),
        date_column="date_day",
    )
)

#=========================================================
#==build html report

report_file = build_model_report(
        model_stage=model_stage,
        model_grain=model_grain,
        target_column=target_column,
        best_model_name=best_model_name,
        comparison_df=comparison_df,
        prediction_df=prediction_df,
        importance_df=importance_df,
        model_df=model_df,
        feature_columns=feature_columns,
        excluded_leakage_columns=(
            excluded_leakage_columns
        ),
        train_rows=len(
            x_train
        ),
        test_rows=len(
            x_test
        ),
        test_start_date=test_start_date,
        output_paths={
            "model": model_file,
            "metrics": metrics_file,
            "predictions": prediction_file,
            "feature_importance": (
                importance_file
            ),
            "feature_list": feature_list_file,
        },
        explanation_result=explanation_result,
        output_path=report_file,
    )

print(f"html report: {report_file}")