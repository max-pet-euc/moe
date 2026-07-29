
import re

import joblib
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import (
    ENGINEERED_DIR,
    OUTPUT_DIR,
)

from src.modelling.feature_selection import select_features
from src.modelling.model_registry import get_models
from src.modelling.trainer import fit_models

#=========================================================
#==config

model_stage = "qs"
# model_stage = "cp"
# model_stage = "rv"

test_days = 90

funnel_order = [
    "us",
    "qs",
    "qc",
    "qe",
    "cp",
    "op",
    "rv",
]

model_targets = {
    "qs": "uncohorted_qs",
    "cp": "uncohorted_cp",
    "rv": "uncohorted_rv",
}

stage_aliases = {
    "us": ["us"],
    "qs": ["qs"],
    "qc": ["qc"],
    "qe": ["qe"],
    "cp": ["cp"],
    "op": ["op"],
    "rv": ["rv", "revenue"],
}

always_excluded_columns = {
    "date_day",
    "date_week",
    "date_month",
    "spend_total",
    "spend_media",
    "spend_digital",
}

input_file = ENGINEERED_DIR / "data_features.csv"

model_output_dir = OUTPUT_DIR / "models"
prediction_output_dir = OUTPUT_DIR / "predictions"

model_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_output_dir.mkdir(
    parents=True,
    exist_ok=True,
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
            <= pd.Timestamp.today().normalize()
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

model_results = []
model_predictions = {}

for model_name, model in trained_models.items():

    prediction = model.predict(
        x_test
    )

    metrics = calculate_metrics(
        actual=y_test,
        predicted=prediction,
    )

    model_results.append(
        {
            "model": model_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "training_seconds": np.nan,
        }
    )

    model_predictions[model_name] = (
        prediction
    )

#=========================================================
#==model comparison

metrics_df = (
    pd.DataFrame(
        model_results
    )
    .sort_values(
        "rmse"
    )
    .reset_index(
        drop=True
    )
)

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
#==predictions

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
    / f"{model_stage}_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False,
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
    / f"{model_stage}_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False,
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
    / f"{model_stage}_features.csv"
)

feature_list_df.to_csv(
    feature_list_file,
    index=False,
)


#=========================================================
#==save metrics

metrics_file = (
    model_output_dir
    / f"{model_stage}_model_comparison.csv"
)

comparison_df.to_csv(
    metrics_file,
    index=False,
)


#=========================================================
#==save best model

model_file = (
    model_output_dir
    / f"{model_stage}_{best_model_name}.joblib"
)

joblib.dump(
    best_model,
    model_file,
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
