"""Weak learner implementations."""

from .base import BinaryWeakLearner, WeakLearnerFitResult
from .classical_linear import WeightedLinearWeakLearner

try:
    from .pqc_binary import WeightedPQCBinaryWeakLearner
except Exception:  # pragma: no cover - optional heavy dependency path
    WeightedPQCBinaryWeakLearner = None  # type: ignore[assignment]

__all__ = ["BinaryWeakLearner", "WeakLearnerFitResult", "WeightedLinearWeakLearner"]
if WeightedPQCBinaryWeakLearner is not None:
    __all__.append("WeightedPQCBinaryWeakLearner")
