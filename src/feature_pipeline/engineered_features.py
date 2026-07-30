# src/feature_pipeline/engineered_features.py

"""
Definitions for engineered marketing features.

This module contains source-column mappings and lightweight
feature-group discovery helpers only.

Calculation logic lives in feature_engineering.py.

Important:
- Digital-platform metrics use the detailed platform dataset.
- Spend rollups use channel-level fields from data_media_inputs.
- Do not combine detailed and channel-level spend in the same rollup,
  because that would double-count the same media spend.
- Digital and offline spend groups are explicitly configured.
- Any remaining source column beginning with "spend_" is treated as
  other spend automatically.
"""

from __future__ import annotations

from collections.abc import Iterable


# =========================================================
# Detailed digital-platform feature groups
# =========================================================

CHANNEL_METRIC_GROUPS: dict[
    str,
    dict[str, str | None],
] = {
    "meta_bof": {
        "sessions": "attribution_meta_meta_bof_last_click_sessions",
        "qs": "digital_meta_meta_bof_qs",
        "cp": "digital_meta_meta_bof_cp",
        "clicks": "digital_meta_meta_bof_clicks",
        "impressions": "digital_meta_meta_bof_impressions",
        "cost": "digital_meta_meta_bof_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "meta_mof": {
        "sessions": "attribution_meta_meta_mof_last_click_sessions",
        "qs": "digital_meta_meta_mof_qs",
        "cp": "digital_meta_meta_mof_cp",
        "clicks": "digital_meta_meta_mof_clicks",
        "impressions": "digital_meta_meta_mof_impressions",
        "cost": "digital_meta_meta_mof_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "meta_tof": {
        "sessions": "attribution_meta_meta_tof_last_click_sessions",
        "qs": "digital_meta_meta_tof_qs",
        "cp": "digital_meta_meta_tof_cp",
        "clicks": "digital_meta_meta_tof_clicks",
        "impressions": "digital_meta_meta_tof_impressions",
        "cost": "digital_meta_meta_tof_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "tiktok": {
        "sessions": "attribution_tiktok_tiktok_paid_last_click_sessions",
        "qs": "digital_tiktok_tiktok_qs",
        "cp": "digital_tiktok_tiktok_cp",
        "clicks": "digital_tiktok_tiktok_clicks",
        "impressions": "digital_tiktok_tiktok_impressions",
        "cost": "digital_tiktok_tiktok_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "search_brand": {
        "sessions": None,
        "qs": "digital_google_search_brand_qs",
        "cp": "digital_google_search_brand_cp",
        "clicks": "digital_google_search_brand_clicks",
        "impressions": "digital_google_search_brand_impressions",
        "cost": "digital_google_search_brand_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "search_nb": {
        "sessions": None,
        "qs": "digital_google_search_nb_qs",
        "cp": "digital_google_search_nb_cp",
        "clicks": "digital_google_search_nb_clicks",
        "impressions": "digital_google_search_nb_impressions",
        "cost": "digital_google_search_nb_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "search_pmax": {
        "sessions": None,
        "qs": "digital_google_search_pmax_qs",
        "cp": "digital_google_search_pmax_cp",
        "clicks": "digital_google_search_pmax_clicks",
        "impressions": "digital_google_search_pmax_impressions",
        "cost": "digital_google_search_pmax_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "search_shopping": {
        "sessions": None,
        "qs": "digital_google_search_shopping_qs",
        "cp": "digital_google_search_shopping_cp",
        "clicks": "digital_google_search_shopping_clicks",
        "impressions": "digital_google_search_shopping_impressions",
        "cost": "digital_google_search_shopping_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "youtube": {
        "sessions": "attribution_google_youtube_paid_last_click_sessions",
        "qs": "digital_google_youtube_qs",
        "cp": "digital_google_youtube_cp",
        "clicks": "digital_google_youtube_clicks",
        "impressions": "digital_google_youtube_impressions",
        "cost": "digital_google_youtube_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
    "reddit": {
        "sessions": None,
        "qs": "digital_reddit_reddit_qs",
        "cp": "digital_reddit_reddit_cp",
        "clicks": "digital_reddit_reddit_clicks",
        "impressions": "digital_reddit_reddit_impressions",
        "cost": "digital_reddit_reddit_spend",
        "market_impressions": None,
        "top_impressions": None,
    },
}


# =========================================================
# Channel-level digital spend rollups
# =========================================================

DIGITAL_SPEND_COLUMNS: dict[str, list[str]] = {
    "meta": ["spend_meta"],
    "tiktok": ["spend_tiktok"],
    "search": [
        "spend_search_bing",
        "spend_search_brand",
        "spend_search_nb",
        "spend_search_pmax",
        "spend_search_shopping",
    ],
    "youtube": ["spend_search_youtube"],
    "reddit": ["spend_reddit"],
    "affiliate": ["spend_affilliate"],
    "display": ["spend_display"],
}


# =========================================================
# Offline spend rollups
# =========================================================

OFFLINE_SPEND_COLUMNS: dict[str, list[str]] = {
    "offline": ["spend_offline"],
}


# =========================================================
# Other spend discovery
# =========================================================

def _flatten_spend_columns(
    spend_groups: dict[str, list[str]],
) -> set[str]:
    """Return all configured source columns from a spend mapping."""

    return {
        column
        for columns in spend_groups.values()
        for column in columns
    }


DEFINED_SPEND_COLUMNS: set[str] = (
    _flatten_spend_columns(DIGITAL_SPEND_COLUMNS)
    | _flatten_spend_columns(OFFLINE_SPEND_COLUMNS)
)


def get_other_spend_columns(
    available_columns: Iterable[str],
) -> dict[str, list[str]]:
    """
    Return unclassified spend columns as individual feature groups.

    Any available column beginning with ``spend_`` is returned unless
    it is already assigned to DIGITAL_SPEND_COLUMNS or
    OFFLINE_SPEND_COLUMNS.

    Examples
    --------
    ``spend_creators`` becomes::

        {"creators": ["spend_creators"]}

    ``spend_partnerships`` becomes::

        {"partnerships": ["spend_partnerships"]}
    """

    other_columns = sorted(
        column
        for column in available_columns
        if (
            column.startswith("spend_")
            and column not in DEFINED_SPEND_COLUMNS
        )
    )

    return {
        column.removeprefix("spend_"): [column]
        for column in other_columns
    }


# =========================================================
# Impression share - overall
# =========================================================

EXTERNAL_METRIC_COLUMNS: dict[str, str] = {
    "impressions": "external_impressions",
    "market_impressions": "external_market_impressions",
    "top_impressions": "external_top_impressions",
}

