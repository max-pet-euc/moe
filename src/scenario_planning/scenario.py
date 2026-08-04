# src/scenario_planning/scenario.py
# SCENARIO_OBJECT_VERSION = "1.0-20260803_2222"

"""
Core scenario-planning objects.

A Scenario combines:
- Commercial targets and approved total budget.
- Growth's channel-level budget plan.
- Metadata used to identify and track the scenario.

Prediction and optimisation outputs will be added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

import pandas as pd


class ScenarioType(StrEnum):
    """How the scenario budget was produced."""

    MANUAL = "manual"
    OPTIMISE_OP = "optimise_op"
    OPTIMISE_CAC = "optimise_cac"


class ScenarioStatus(StrEnum):
    """Current lifecycle state of a scenario."""

    DRAFT = "draft"
    VALIDATED = "validated"
    PREDICTED = "predicted"
    APPROVED = "approved"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class CommercialTargets:
    """Commercial targets and approved media budget."""

    date_month: pd.Timestamp
    target_uncohorted_qs: float
    target_uncohorted_op: float
    target_budget_media: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "date_month",
            pd.Timestamp(
                self.date_month
            ).normalize(),
        )

@dataclass(frozen=True)
class GrowthBudgetPlan:
    """Growth's monthly budget allocation."""

    date_month: pd.Timestamp
    budget_total: float
    budget_media: float
    channel_budgets: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "date_month",
            pd.Timestamp(
                self.date_month
            ).normalize(),
        )

        object.__setattr__(
            self,
            "budget_total",
            float(self.budget_total),
        )

        object.__setattr__(
            self,
            "budget_media",
            float(self.budget_media),
        )

        object.__setattr__(
            self,
            "channel_budgets",
            {
                str(channel): float(value)
                for channel, value
                in self.channel_budgets.items()
            },
        )

    @property
    def allocated_budget_total(
        self,
    ) -> float:
        """Sum all detailed channel budgets."""

        return float(
            sum(
                self.channel_budgets.values()
            )
        )

@dataclass
class Scenario:
    """One complete monthly planning scenario."""

    scenario_id: str
    scenario_name: str
    commercial_targets: CommercialTargets
    growth_budget_plan: GrowthBudgetPlan

    scenario_type: ScenarioType = (
        ScenarioType.MANUAL
    )

    status: ScenarioStatus = (
        ScenarioStatus.DRAFT
    )

    flighting_profile: str = (
        "standard_monthly"
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[
        str,
        str | float | int | bool,
    ] = field(
        default_factory=dict
    )

    @property
    def date_month(
        self,
    ) -> pd.Timestamp:
        """Planning month represented by this scenario."""

        return (
            self.commercial_targets
            .date_month
        )

    @property
    def days(
        self,
    ) -> int:
        """Number of calendar days in the planning month."""

        return int(
            self.date_month.days_in_month
        )

    @property
    def budget_total(
        self,
    ) -> float:
        """Growth's stated total budget."""

        return float(
            self.growth_budget_plan
            .budget_total
        )

    @property
    def budget_media(
        self,
    ) -> float:
        """Growth's stated media budget."""

        return float(
            self.growth_budget_plan
            .budget_media
        )

    @property
    def target_budget_media(
        self,
    ) -> float:
        """Commercial's approved media budget."""

        return float(
            self.commercial_targets
            .target_budget_media
        )

    @property
    def allocated_budget_total(
        self,
    ) -> float:
        """Sum of detailed Growth budget lines."""

        return (
            self.growth_budget_plan
            .allocated_budget_total
        )

    @property
    def budget_total_variance(
        self,
    ) -> float:
        """Detailed allocation minus stated total."""

        return (
            self.allocated_budget_total
            - self.budget_total
        )

    @property
    def media_target_variance(
        self,
    ) -> float:
        """Growth media budget minus Commercial target."""

        return (
            self.budget_media
            - self.target_budget_media
        )

    def to_summary_dict(
        self,
    ) -> dict[str, object]:
        """Return a flat scenario summary for CSV or reporting."""

        output: dict[str, object] = {
            "scenario_id": (
                self.scenario_id
            ),
            "scenario_name": (
                self.scenario_name
            ),
            "scenario_type": (
                self.scenario_type.value
            ),
            "status": (
                self.status.value
            ),
            "flighting_profile": (
                self.flighting_profile
            ),
            "date_month": (
                self.date_month.date()
            ),
            "days": (
                self.days
            ),
            "target_uncohorted_qs": (
                self.commercial_targets
                .target_uncohorted_qs
            ),
            "target_uncohorted_op": (
                self.commercial_targets
                .target_uncohorted_op
            ),
            "target_budget_media": (
                self.target_budget_media
            ),
            "budget_media": (
                self.budget_media
            ),
            "budget_total": (
                self.budget_total
            ),
            "allocated_budget_total": (
                self.allocated_budget_total
            ),
            "budget_total_variance": (
                self.budget_total_variance
            ),
            "media_target_variance": (
                self.media_target_variance
            ),
            "created_at": (
                self.created_at
                .isoformat()
            ),
        }

        output.update(
            self.growth_budget_plan
            .channel_budgets
        )

        output.update(
            {
                f"metadata_{key}": value
                for key, value
                in self.metadata.items()
            }
        )

        return output
