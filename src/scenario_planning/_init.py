# src/scenario_planning/__init__.py
# SCENARIO_PLANNING_VERSION = "1.0-20260803_2222"

"""Scenario planning objects and input validation."""

from .scenario import (
    CommercialTargets,
    GrowthBudgetPlan,
    Scenario,
    ScenarioStatus,
    ScenarioType,
)
from .validate_inputs import (
    PlanningInputValidationError,
    PlanningValidationResult,
    build_scenarios,
    load_and_validate_planning_inputs,
)

__all__ = [
    "CommercialTargets",
    "GrowthBudgetPlan",
    "PlanningInputValidationError",
    "PlanningValidationResult",
    "Scenario",
    "ScenarioStatus",
    "ScenarioType",
    "build_scenarios",
    "load_and_validate_planning_inputs",
]
