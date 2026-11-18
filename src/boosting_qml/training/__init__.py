"""Training utilities for EM-style PQC experiments."""

from __future__ import annotations

from .convergence import (
    BEST_LOSS_NO_IMPROVEMENT_DEFINITION,
    BestLossNoImprovementConfig,
    BestLossNoImprovementDecision,
    BestLossNoImprovementTracker,
    WindowedPlateauConfig,
    WindowedPlateauDecision,
    WindowedPlateauTracker,
)

_EM_EXPORTS = {
    "StaticConfig",
    "BEST_LOSS_NO_IMPROVEMENT_DEFINITION",
    "BestLossNoImprovementConfig",
    "BestLossNoImprovementDecision",
    "BestLossNoImprovementTracker",
    "accumulate_gradients_scan",
    "accumulate_parameter_shift_gradients_scan",
    "evaluate_batch",
    "train_batch",
    "train_batch_parameter_shift_binomial",
}

__all__ = [
    "StaticConfig",
    "WindowedPlateauConfig",
    "WindowedPlateauDecision",
    "WindowedPlateauTracker",
    "accumulate_gradients_scan",
    "accumulate_parameter_shift_gradients_scan",
    "evaluate_batch",
    "train_batch",
    "train_batch_parameter_shift_binomial",
]


def __getattr__(name: str):
    """Lazily import EM/JAX-heavy helpers only when requested."""

    if name in _EM_EXPORTS:
        from . import em as _em

        value = getattr(_em, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
