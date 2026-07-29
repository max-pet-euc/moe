"""
Reporting helpers.
"""

from pathlib import Path


def ensure_directory(
    path,
):

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_dataframe(
    dataframe,
    output_path,
):

    ensure_directory(
        Path(output_path).parent,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )