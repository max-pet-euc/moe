"""Select valid features for each modelling target."""

import pandas as pd


LEAKAGE_COLUMNS = {
    "qs": ["cp", "op", "revenue"],
    "cp": ["op", "revenue"],
    "revenue": [],
}


def prepare_target_data(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    excluded = [target, "date_day"] + LEAKAGE_COLUMNS.get(target, [])

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return df[feature_columns], df[target]
