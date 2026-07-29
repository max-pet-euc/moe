"""
Evaluate trained models.
"""

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_models(
    trained_models,
    x_test,
    y_test,
):

    results = []

    for model_name, model in trained_models.items():

        predictions = model.predict(x_test)

        results.append({

            "model": model_name,

            "r2": r2_score(
                y_test,
                predictions,
            ),

            "rmse": (
                mean_squared_error(
                    y_test,
                    predictions,
                ) ** 0.5
            ),

            "mae": mean_absolute_error(
                y_test,
                predictions,
            ),

        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "rmse",
        )
        .reset_index(drop=True)
    )