"""
Evaluate trained models.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]


def evaluate_models(
    trained_models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:

    results = []
    predictions = {}

    for model_name, model in trained_models.items():

        model_prediction = model.predict(
            x_test
        )

        results.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(
                    y_test,
                    model_prediction,
                ),
                "rmse": (
                    mean_squared_error(
                        y_test,
                        model_prediction,
                    )
                    ** 0.5
                ),
                "r2": r2_score(
                    y_test,
                    model_prediction,
                ),
            }
        )

        predictions[
            model_name
        ] = model_prediction

    metrics = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "rmse"
        )
        .reset_index(
            drop=True
        )
    )

    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
    )