"""Windowed plateau stopping for optimizer-level convergence.

The criterion compares the first and second halves of the most recent loss
window.  A run is treated as plateaued when the two sample means are closer than
``scale`` times the standard error of the second-half mean.  This is a
heuristic, not a statistical test: minibatch losses are correlated, shot noise
can be non-Gaussian, and fixed Monte Carlo trajectories make the objective a
sample-average approximation.  The goal is to stop optimization after the loss
has flattened while keeping fixed-size JAX training chunks reusable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


WINDOWED_PLATEAU_CONVERGENCE_DEFINITION = (
    "windowed_half_mean_gap_le_scale_times_second_half_sem.v1"
)
BEST_LOSS_NO_IMPROVEMENT_DEFINITION = "best_loss_no_improvement_patience.v1"


@dataclass(frozen=True)
class WindowedPlateauConfig:
    """Configuration for loss-window plateau detection."""

    enabled: bool = False
    window_size: int = 50
    min_steps: int = 50
    patience: int = 1
    scale: float = 1.0

    def validate(self) -> None:
        if self.window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {self.window_size}.")
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be positive, got {self.min_steps}.")
        if self.patience < 1:
            raise ValueError(f"patience must be positive, got {self.patience}.")
        if self.scale < 0.0:
            raise ValueError(f"scale must be nonnegative, got {self.scale}.")


@dataclass(frozen=True)
class WindowedPlateauDecision:
    """One plateau-check decision."""

    step: int
    loss: float
    eligible: bool
    hit: bool
    converged: bool
    consecutive_hits: int
    mu_first: float | None = None
    mu_second: float | None = None
    sigma_second: float | None = None
    sem_second: float | None = None
    threshold: float | None = None
    mean_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WindowedPlateauTracker:
    """Mutable tracker for the windowed plateau rule."""

    def __init__(self, config: WindowedPlateauConfig):
        config.validate()
        self.config = config
        self.loss_history: list[float] = []
        self.consecutive_hits = 0
        self.converged = False
        self.stop_step: int | None = None
        self.last_decision: WindowedPlateauDecision | None = None

    def update(self, loss: float, *, step: int) -> WindowedPlateauDecision:
        value = float(loss)
        self.loss_history.append(value)

        decision = self._check(value, int(step))
        self.last_decision = decision
        if decision.converged and self.stop_step is None:
            self.converged = True
            self.stop_step = int(step)
        return decision

    def _check(self, loss: float, step: int) -> WindowedPlateauDecision:
        if not self.config.enabled:
            return WindowedPlateauDecision(
                step=step,
                loss=loss,
                eligible=False,
                hit=False,
                converged=False,
                consecutive_hits=self.consecutive_hits,
            )

        eligible = step >= self.config.min_steps and len(self.loss_history) >= self.config.window_size
        if not eligible:
            self.consecutive_hits = 0
            return WindowedPlateauDecision(
                step=step,
                loss=loss,
                eligible=False,
                hit=False,
                converged=False,
                consecutive_hits=self.consecutive_hits,
            )

        window = np.asarray(self.loss_history[-self.config.window_size :], dtype=np.float64)
        if not np.all(np.isfinite(window)):
            self.consecutive_hits = 0
            return WindowedPlateauDecision(
                step=step,
                loss=loss,
                eligible=True,
                hit=False,
                converged=False,
                consecutive_hits=self.consecutive_hits,
            )

        split = self.config.window_size // 2
        first = window[:split]
        second = window[split:]
        mu_first = float(np.mean(first))
        mu_second = float(np.mean(second))
        sigma_second = float(np.std(second, ddof=1 if second.size > 1 else 0))
        sem_second = float(sigma_second / np.sqrt(max(1, second.size)))
        threshold = float(self.config.scale * sem_second)
        mean_gap = float(abs(mu_first - mu_second))
        hit = bool(mean_gap <= threshold)

        if hit:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        converged = bool(self.consecutive_hits >= self.config.patience)
        return WindowedPlateauDecision(
            step=step,
            loss=loss,
            eligible=True,
            hit=hit,
            converged=converged,
            consecutive_hits=int(self.consecutive_hits),
            mu_first=mu_first,
            mu_second=mu_second,
            sigma_second=sigma_second,
            sem_second=sem_second,
            threshold=threshold,
            mean_gap=mean_gap,
        )

    def summary(self, *, max_history_tail: int = 10) -> dict[str, Any]:
        tail = self.loss_history[-max(0, int(max_history_tail)) :] if max_history_tail else []
        return {
            "convergence_definition": WINDOWED_PLATEAU_CONVERGENCE_DEFINITION,
            "enabled": bool(self.config.enabled),
            "window_size": int(self.config.window_size),
            "min_steps": int(self.config.min_steps),
            "patience": int(self.config.patience),
            "scale": float(self.config.scale),
            "converged": bool(self.converged),
            "stop_step": None if self.stop_step is None else int(self.stop_step),
            "loss_history_length": int(len(self.loss_history)),
            "loss_history_tail": [float(v) for v in tail],
            "consecutive_hits": int(self.consecutive_hits),
            "last_decision": None if self.last_decision is None else self.last_decision.to_dict(),
        }


@dataclass(frozen=True)
class BestLossNoImprovementConfig:
    """Epoch-wise best-loss early stopping with no implicit warm-up floor."""

    enabled: bool = True
    patience: int = 20
    min_delta: float = 0.0
    check_frequency_epochs: int = 1

    def validate(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience must be positive, got {self.patience}.")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be nonnegative, got {self.min_delta}.")
        if self.check_frequency_epochs != 1:
            raise ValueError(
                "best_loss_no_improvement_patience.v1 must be checked every epoch."
            )


@dataclass(frozen=True)
class BestLossNoImprovementDecision:
    epoch: int
    current_epoch_loss: float
    improved: bool
    best_epoch_loss: float
    best_epoch: int
    epochs_without_improvement: int
    should_stop: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BestLossNoImprovementTracker:
    """Stateful implementation of the final symmetric stopping rule."""

    def __init__(
        self,
        config: BestLossNoImprovementConfig,
        *,
        best_epoch_loss: float | None = None,
        best_epoch: int = 0,
        epochs_without_improvement: int = 0,
        current_epoch: int = 0,
    ):
        config.validate()
        self.config = config
        self.best_epoch_loss = (
            None if best_epoch_loss is None else float(best_epoch_loss)
        )
        self.best_epoch = int(best_epoch)
        self.epochs_without_improvement = int(epochs_without_improvement)
        self.current_epoch = int(current_epoch)
        self.last_decision: BestLossNoImprovementDecision | None = None

    def update(
        self,
        current_epoch_loss: float,
        *,
        epoch: int,
    ) -> BestLossNoImprovementDecision:
        value = float(current_epoch_loss)
        current_epoch = int(epoch)
        if current_epoch != self.current_epoch + 1:
            raise ValueError(
                "Epochs must be consecutive when updating early stopping: "
                f"current={self.current_epoch}, received={current_epoch}."
            )
        if not np.isfinite(value):
            improved = False
        else:
            improved = bool(
                self.best_epoch_loss is None
                or value < self.best_epoch_loss - float(self.config.min_delta)
            )
        if improved:
            self.best_epoch_loss = value
            self.best_epoch = current_epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        self.current_epoch = current_epoch
        should_stop = bool(
            self.config.enabled
            and self.epochs_without_improvement >= self.config.patience
        )
        decision = BestLossNoImprovementDecision(
            epoch=current_epoch,
            current_epoch_loss=value,
            improved=improved,
            best_epoch_loss=float(self.best_epoch_loss),
            best_epoch=int(self.best_epoch),
            epochs_without_improvement=int(self.epochs_without_improvement),
            should_stop=should_stop,
        )
        self.last_decision = decision
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_stopping_definition": BEST_LOSS_NO_IMPROVEMENT_DEFINITION,
            "enabled": bool(self.config.enabled),
            "fixed_min_epochs": None,
            "early_stopping_patience_epochs": int(self.config.patience),
            "early_stopping_check_frequency_epochs": int(
                self.config.check_frequency_epochs
            ),
            "early_stopping_min_delta": float(self.config.min_delta),
            "best_epoch_loss": self.best_epoch_loss,
            "best_epoch": int(self.best_epoch),
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "current_epoch": int(self.current_epoch),
            "last_decision": (
                None if self.last_decision is None else self.last_decision.to_dict()
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "BestLossNoImprovementTracker":
        if (
            state.get("training_stopping_definition")
            != BEST_LOSS_NO_IMPROVEMENT_DEFINITION
        ):
            raise ValueError("Early-stopping resume state has an incompatible definition.")
        return cls(
            BestLossNoImprovementConfig(
                enabled=bool(state.get("enabled", True)),
                patience=int(state["early_stopping_patience_epochs"]),
                min_delta=float(state["early_stopping_min_delta"]),
                check_frequency_epochs=int(
                    state["early_stopping_check_frequency_epochs"]
                ),
            ),
            best_epoch_loss=state.get("best_epoch_loss"),
            best_epoch=int(state.get("best_epoch", 0)),
            epochs_without_improvement=int(
                state.get("epochs_without_improvement", 0)
            ),
            current_epoch=int(state.get("current_epoch", 0)),
        )
