"""
Train models.
"""

from sklearn.base import clone


def fit_models(
    models: dict,
    x_train,
    y_train,
) -> dict:

    trained_models = {}

    for model_name, model in models.items():

        fitted_model = clone(model)

        fitted_model.fit(
            x_train,
            y_train,
        )

        trained_models[model_name] = fitted_model

    return trained_models