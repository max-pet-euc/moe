# src/scenario_planning/__init__.py
# SCENARIO_PLANNING_VERSION = "1.1-20260804_1147"

"""Scenario planning objects, validation and daily-plan building."""

from .build_daily_plan import (
    DailyPlanBuildError,
    DailyPlanBuildResult,
    allocate_monthly_value_evenly,
    build_and_save_daily_plans,
    build_daily_plans,
    build_daily_scenario_plan,
)
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
    "DailyPlanBuildError",
    "DailyPlanBuildResult",
    "GrowthBudgetPlan",
    "PlanningInputValidationError",
    "PlanningValidationResult",
    "Scenario",
    "ScenarioStatus",
    "ScenarioType",
    "allocate_monthly_value_evenly",
    "build_and_save_daily_plans",
    "build_daily_plans",
    "build_daily_scenario_plan",
    "build_scenarios",
    "load_and_validate_planning_inputs",
]