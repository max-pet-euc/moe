"""Create the canonical engineered feature dataset."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the merged daily dataset."""
    output = df.copy()

    # Placeholder:
    # - joins
    # - adstock
    # - lags
    # - rolling windows
    # - log transforms
    # - interaction terms

    return output
