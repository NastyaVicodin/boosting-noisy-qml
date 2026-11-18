"""Typed configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


def scalar_tag(value: float, *, prefix: str = '') -> str:
    """Return a filesystem-friendly tag for a scalar hyperparameter."""

    text = f'{float(value):.8g}'
    text = text.replace('-', 'm').replace('+', '')
    text = text.replace('.', 'p')
    return f'{prefix}{text}'


@dataclass(frozen=True)
class Phase0ExperimentConfig:
    """Configuration for classical Phase-0 scaling comparisons."""

    seed: int = 7
    n_samples: int = 320
    n_features: int = 5
    test_fraction: float = 0.3
    label_noise: float = 0.05
    n_rounds: int = 30
    margin_threshold: float = 0.2
    target_margin_loss: float = 0.1
    ridge_lambda: float = 1e-2
    score_scale: float = 1.0
    eta_grid: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2)
    margin_grid: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EMRunArtifactConfig:
    """How to discover EM independent-run artifacts from disk."""

    artifact_dir: str
    glob_pattern: str = '*.json'
    recursive: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_dir:
            raise ValueError('artifact_dir must be a non-empty path.')
        if not self.glob_pattern:
            raise ValueError('glob_pattern must be non-empty.')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EMTargetAggregationConfig:
    """Risk-aggregation knobs for EM summary targets fed into boosting bounds."""

    target_name: str = 'margin_loss'
    strategy: str = 'pathwise_pessimistic'
    percentile_q: float = 0.9
    cvar_tail_fraction: float = 0.2
    lower_is_better: bool | None = None

    def __post_init__(self) -> None:
        valid_targets = {'margin_loss', 'classification_error', 'mean_margin', 'test_accuracy', 'validation_accuracy'}
        valid_strategies = {'pathwise_pessimistic', 'mean', 'median', 'percentile', 'cvar'}
        if self.target_name not in valid_targets:
            raise ValueError(f'target_name must be one of {sorted(valid_targets)}.')
        if self.strategy not in valid_strategies:
            raise ValueError(f'strategy must be one of {sorted(valid_strategies)}.')
        if not (0.0 < self.percentile_q < 1.0):
            raise ValueError('percentile_q must lie in (0, 1).')
        if not (0.0 < self.cvar_tail_fraction <= 1.0):
            raise ValueError('cvar_tail_fraction must lie in (0, 1].')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoundBudgetBoundsConfig:
    """Controls smoothed lower-bound families used in per-round T diagnostics."""

    lower_percentile_q: float = 0.1
    lower_cvar_tail_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not (0.0 < self.lower_percentile_q < 1.0):
            raise ValueError('lower_percentile_q must lie in (0, 1).')
        if not (0.0 < self.lower_cvar_tail_fraction <= 1.0):
            raise ValueError('lower_cvar_tail_fraction must lie in (0, 1].')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BoostingStopTargetConfig:
    """Explicit stopping target used for PEC-vs-boosting comparisons."""

    target_margin_loss: float
    target_validation_accuracy: Optional[float]
    validation_accuracy_tolerance: float = 0.02
    min_rounds: int = 1
    patience: int = 1
    use_planned_round_budget: bool = True
    stop_mode: str = 'match_em'
    train_margin_mu: Optional[float] = None
    train_margin_target: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_margin_loss <= 1.0):
            raise ValueError('target_margin_loss must lie in [0, 1].')
        valid_stop_modes = {
            'match_em',
            'fixed_validation_accuracy',
            'fixed_train_margin_loss',
        }
        if self.stop_mode not in valid_stop_modes:
            raise ValueError(f'stop_mode must be one of {sorted(valid_stop_modes)}.')
        if self.stop_mode == 'fixed_train_margin_loss':
            if self.train_margin_mu is None or not np.isfinite(self.train_margin_mu) or self.train_margin_mu <= 0.0:
                raise ValueError(
                    "train_margin_mu must be finite and positive when "
                    "stop_mode='fixed_train_margin_loss'."
                )
            if self.train_margin_target is None or not (0.0 <= self.train_margin_target <= 1.0):
                raise ValueError(
                    "train_margin_target must lie in [0, 1] when "
                    "stop_mode='fixed_train_margin_loss'."
                )
            if self.target_validation_accuracy is not None:
                raise ValueError(
                    "target_validation_accuracy must be omitted for "
                    "stop_mode='fixed_train_margin_loss'."
                )
            if float(self.target_margin_loss) != float(self.train_margin_target):
                raise ValueError(
                    "target_margin_loss and train_margin_target must be identical "
                    "for fixed_train_margin_loss."
                )
            if self.min_rounds != 1 or self.patience != 1:
                raise ValueError(
                    "fixed_train_margin_loss stops immediately after the first "
                    "accepted prefix attaining the target; min_rounds and patience "
                    "must both equal 1."
                )
        else:
            if self.train_margin_mu is not None or self.train_margin_target is not None:
                raise ValueError(
                    "train-margin fields are only valid for "
                    "stop_mode='fixed_train_margin_loss'."
                )
            if self.target_validation_accuracy is None or not (
                0.0 <= self.target_validation_accuracy <= 1.0
            ):
                raise ValueError('target_validation_accuracy must lie in [0, 1].')
        if not (0.0 <= self.validation_accuracy_tolerance <= 1.0):
            raise ValueError('validation_accuracy_tolerance must lie in [0, 1].')
        if self.min_rounds <= 0:
            raise ValueError('min_rounds must be positive.')
        if self.patience <= 0:
            raise ValueError('patience must be positive.')

    def to_dict(self) -> dict:
        return asdict(self)
