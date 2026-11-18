"""Theory-side analysis utilities."""

from .round_scaling import (
    RunningConvergenceDiagnostics,
    RunningRoundBudgetEstimate,
    RoundScalingPoint,
    compare_round_scaling_grid,
    estimate_equilibrium_margin,
    estimate_round_budget_from_cumulative_edge,
    estimate_running_convergence_diagnostics,
    extract_round_budget_bound_families,
    estimate_running_round_budget_adaboost_r,
    estimate_running_round_budget_real_adaboost,
    estimate_round_budget_adaboost_r,
    estimate_round_budget_real_adaboost,
)

__all__ = [
    "RunningConvergenceDiagnostics",
    "RunningRoundBudgetEstimate",
    "RoundScalingPoint",
    "compare_round_scaling_grid",
    "estimate_equilibrium_margin",
    "estimate_round_budget_from_cumulative_edge",
    "estimate_running_convergence_diagnostics",
    "extract_round_budget_bound_families",
    "estimate_running_round_budget_adaboost_r",
    "estimate_running_round_budget_real_adaboost",
    "estimate_round_budget_adaboost_r",
    "estimate_round_budget_real_adaboost",
]
