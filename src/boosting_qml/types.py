"""Typed containers shared across the codebase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

Array = np.ndarray


@dataclass
class BinaryDataset:
    """Binary dataset with labels constrained to {-1, +1}."""

    X: Array
    y: Array

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.int8)

        if self.X.ndim != 2:
            raise ValueError('X must be a rank-2 array of shape [n_samples, n_features].')
        if self.y.ndim != 1:
            raise ValueError('y must be a rank-1 array of shape [n_samples].')
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('X and y must have the same number of samples.')
        if not np.all(np.isin(self.y, [-1, 1])):
            raise ValueError('Binary labels must be encoded as -1 or +1.')

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass(frozen=True)
class BinaryDatasetSplit:
    """Train/validation/test split for binary datasets."""

    train: BinaryDataset
    validation: Optional[BinaryDataset]
    test: BinaryDataset


@dataclass
class BoostingRoundResult:
    """Per-round diagnostics for binary boosting algorithms."""

    round_index: int
    edge_t: float
    boost_coef_t: float
    train_scores: Array
    train_ensemble_score: Array
    example_weights_before_update: Array
    example_weights_after_update: Array
    empirical_train_error: float
    empirical_train_margin_loss: float
    empirical_train_logistic_margin_loss: Optional[float] = None
    weighted_surrogate_loss_t: Optional[float] = None
    surrogate_smooth_hinge_loss_t: Optional[float] = None
    surrogate_edge_lower_bound_t: Optional[float] = None
    surrogate_sigma_t: Optional[float] = None
    validation_scores: Optional[Array] = None
    validation_ensemble_score: Optional[Array] = None
    validation_accuracy: Optional[float] = None
    empirical_validation_error: Optional[float] = None
    empirical_validation_margin_loss: Optional[float] = None
    validation_monitor_accuracy: Optional[float] = None
    empirical_validation_monitor_error: Optional[float] = None
    empirical_validation_monitor_margin_loss: Optional[float] = None
    validation_monitor_triggered: bool = False
    validation_confirmation_attempted: bool = False
    validation_confirmation_policy: Optional[str] = None
    intra_round_target_confirmed: bool = False
    intra_round_target_epoch: Optional[int] = None
    intra_round_target_check_reason: Optional[str] = None
    test_scores: Optional[Array] = None
    test_accuracy: Optional[float] = None
    empirical_test_margin_loss: Optional[float] = None
    test_evaluation_policy: Optional[str] = None
    test_evaluation_is_final: bool = False
    test_evaluation_is_diagnostic: bool = False
    final_test_selection_reason: Optional[str] = None
    final_test_shot_accounting: Optional[dict[str, Any]] = None
    final_test_wall_clock_seconds: float = 0.0
    h_t_star: Optional[float] = None
    shot_budget_consumed: int = 0
    shot_accounting: Optional[dict[str, Any]] = None
    wall_clock_seconds: float = 0.0
    rng_seed: Optional[int] = None
    rng_stream_seeds: Optional[dict[str, int]] = None
    weak_learner_epochs_run: Optional[int] = None
    weak_learner_converged: Optional[bool] = None
    weak_learner_convergence_diagnostics: Optional[dict[str, Any]] = None
    train_margin_mu: Optional[float] = None
    train_margin_target: Optional[float] = None
    train_margin_target_reached: Optional[bool] = None
    train_margin_candidate_reached: Optional[bool] = None
    train_margin_confirmation_attempted: bool = False
    train_margin_confirmation_policy: Optional[str] = None
    train_margin_confirmation_loss: Optional[float] = None
    train_margin_confirmation_reached: Optional[bool] = None
    train_margin_confirmation_ensemble_score: Optional[Array] = None
    train_margin_confirmation_shot_accounting: Optional[dict[str, Any]] = None
    deterministic_train_margin_loss: Optional[float] = None
    ensemble_patience_improved: Optional[bool] = None
    ensemble_patience_rounds_since_improvement: Optional[int] = None


@dataclass(frozen=True)
class BoostingStopSummary:
    """Outcome of an explicit stopping policy for one boosting run."""

    target_reached: bool
    stop_round_index: int
    stop_reason: str
    first_hit_round_index: Optional[int] = None
    first_hit_weak_learner_epoch: Optional[int] = None
    target_margin_loss: Optional[float] = None
    target_validation_accuracy: Optional[float] = None
    validation_accuracy_tolerance: Optional[float] = None
    min_rounds: Optional[int] = None
    patience: Optional[int] = None
    hit_train_margin_loss: Optional[float] = None
    hit_validation_accuracy: Optional[float] = None
    hit_test_accuracy: Optional[float] = None
    stop_mode: str = 'match_em'
    train_margin_mu: Optional[float] = None
    train_margin_target: Optional[float] = None
    ensemble_patience_rounds: Optional[int] = None
    ensemble_patience_min_delta: Optional[float] = None
    ensemble_best_round_index: Optional[int] = None
    ensemble_best_train_margin_loss: Optional[float] = None
    ensemble_rounds_since_improvement: Optional[int] = None


@dataclass(frozen=True)
class EMRunSummary:
    """Canonical per-run summary extracted from one EM artifact file."""

    run_id: str
    source_path: str
    target_margin_loss: float
    target_classification_error: float
    target_mean_margin: float
    reference_test_accuracy: float
    rho: float
    margin_threshold: Optional[float]
    margin_tag: Optional[str]
    n_shots_eff: int
    pec_gamma_sq: float
    n_shots_raw_pec: float
    reference_validation_accuracy: Optional[float] = None
    dataset_kind: Optional[str] = None
    n_qubits: Optional[int] = None
    n_layers: Optional[int] = None
    n_train_samples: Optional[int] = None
    n_validation_samples: Optional[int] = None
    n_test_samples: Optional[int] = None
    n_epochs_run: Optional[int] = None
    shot_accounting: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class EMTargetAggregationSummary:
    """Risk-aggregated EM target statistics across independent runs."""

    target_name: str
    n_runs: int
    lower_is_better: bool
    pathwise_pessimistic: float
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    percentile: float
    percentile_q: float
    cvar: float
    cvar_tail_fraction: float


@dataclass(frozen=True)
class RoundBudgetBoundFamily:
    """Per-round multi-bound view of edge and T estimates."""

    round_index: int
    edge_pathwise_min_t: float
    edge_lcb_pathwise_min_t: float
    edge_effective_t: float
    edge_percentile_t: float
    edge_cvar_t: float
    estimated_round_budget_pathwise_t: float
    estimated_round_budget_lcb_t: float
    estimated_round_budget_effective_t: float
    estimated_round_budget_percentile_t: float
    estimated_round_budget_cvar_t: float
    surrogate_edge_lower_bound_t: Optional[float] = None
    surrogate_edge_pathwise_min_t: Optional[float] = None
    estimated_round_budget_surrogate_pathwise_t: Optional[float] = None
