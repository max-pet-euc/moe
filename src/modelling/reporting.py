"""
Reporting helpers.
"""

from pathlib import Path

import joblib
import pandas as pd


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


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path,
) -> Path:

    output_path = Path(
        output_path
    )

    ensure_directory(
        output_path.parent
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_model(
    model,
    output_path,
) -> Path:

    output_path = Path(
        output_path
    )

    ensure_directory(
        output_path.parent
    )

    joblib.dump(
        model,
        output_path,
    )

    return output_path
