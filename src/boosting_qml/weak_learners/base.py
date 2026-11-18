"""Weak learner protocol for binary boosting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

import numpy as np

from boosting_qml.types import Array, BinaryDataset


@dataclass(frozen=True)
class WeakLearnerFitResult:
    """Fit diagnostics from a weak learner training step."""

    weighted_surrogate_loss_t: float
    epochs_run: Optional[int] = None
    converged: Optional[bool] = None
    convergence_diagnostics: Optional[Mapping[str, Any]] = None
    target_confirmed: bool = False
    target_confirmation: Optional[Mapping[str, Any]] = None


class BinaryWeakLearner(Protocol):
    """Protocol implemented by binary weak learners."""

    def fit_weighted(
        self,
        dataset: BinaryDataset,
        example_weights: Array,
        rng: np.random.Generator,
        epoch_callback: Optional[
            Callable[[Any, int, str], Mapping[str, Any] | None]
        ] = None,
    ) -> WeakLearnerFitResult:
        ...

    def predict_score(
        self,
        x: Array,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        ...

    def predict_label(
        self,
        x: Array,
        threshold: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BinaryWeakLearner":
        ...
