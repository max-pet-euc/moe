"""
Model definitions for the Marketing Optimisation Engine.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def get_models(random_state: int = 42) -> dict:

    return {

        "linear": LinearRegression(),

        "ridge": Ridge(
            alpha=1.0,
            random_state=random_state,
        ),

        "lasso": Lasso(
            alpha=0.001,
            random_state=random_state,
            max_iter=10000,
        ),

        "elastic_net": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            random_state=random_state,
            max_iter=10000,
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),

    }