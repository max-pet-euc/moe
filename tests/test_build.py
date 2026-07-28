"""
Basic smoke test for the feature build.
"""

from src.build_features import build_features

def test_build_runs():
    features = build_features()
    assert len(features) > 0
    assert "date_day" in features.columns
