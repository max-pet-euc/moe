# src/scenario_planning/scenario_feature_settings.py
# SCENARIO_FEATURE_SETTINGS_VERSION = "1.0-20260804_1409"

"""
Mappings used to convert Growth's aggregate monthly budget plan into
the detailed daily columns used by the MOE feature dataset.

Planning budgets are intentionally aggregate:
- Search
- YouTube
- Meta
- TikTok
- Reddit
- TV
- VOD
- OOH
- Creator

The model uses more detailed Search and Meta fields. Their planned
budgets are split using recent historical spend shares.
"""

from __future__ import annotations


# =========================================================
# Planning budget -> wide media-input feature
# =========================================================

DIRECT_MEDIA_BUDGET_MAPPINGS: dict[str, str] = {
    "daily_budget_youtube": "spend_search_youtube",
    "daily_budget_meta": "spend_meta",
    "daily_budget_tiktok": "spend_tiktok",
    "daily_budget_reddit": "spend_reddit",
    "daily_budget_tv": "spend_tv",
    "daily_budget_vod": "spend_vod",
    "daily_budget_ooh": "spend_ooh",
    "daily_budget_creator": "spend_creators",
}


# =========================================================
# Aggregate Search budget split
# =========================================================

SEARCH_MEDIA_COLUMNS: list[str] = [
    "spend_search_bing",
    "spend_search_brand",
    "spend_search_nb",
    "spend_search_pmax",
    "spend_search_shopping",
]


SEARCH_DIGITAL_SPEND_COLUMNS: dict[str, str] = {
    "spend_search_brand": (
        "digital_google_search_brand_spend"
    ),
    "spend_search_nb": (
        "digital_google_search_nb_spend"
    ),
    "spend_search_pmax": (
        "digital_google_search_pmax_spend"
    ),
    "spend_search_shopping": (
        "digital_google_search_shopping_spend"
    ),
}


# =========================================================
# Aggregate Meta budget split
# =========================================================

META_DIGITAL_SPEND_COLUMNS: list[str] = [
    "digital_meta_meta_bof_spend",
    "digital_meta_meta_mof_spend",
    "digital_meta_meta_tof_spend",
]


# =========================================================
# Direct digital-platform spend mappings
# =========================================================

DIRECT_DIGITAL_SPEND_MAPPINGS: dict[str, str] = {
    "daily_budget_youtube": (
        "digital_google_youtube_spend"
    ),
    "daily_budget_tiktok": (
        "digital_tiktok_tiktok_spend"
    ),
    "daily_budget_reddit": (
        "digital_reddit_reddit_spend"
    ),
}


# =========================================================
# Historical assumption configuration
# =========================================================

HISTORICAL_LOOKBACK_DAYS = 28

MINIMUM_HISTORY_DAYS = 7

DEFAULT_SHARE_WHEN_NO_HISTORY = 0.0

ASSUMPTION_TOLERANCE = 0.01


# =========================================================
# Columns that must never be copied into a future scenario
# =========================================================

EXCLUDED_FUTURE_PREFIXES: tuple[str, ...] = (
    "attribution_",
    "uncohorted_",
    "cohorted_",
)

EXCLUDED_FUTURE_COLUMNS: set[str] = {
    "spend_total",
    "spend_media",
    "spend_digital",
}
