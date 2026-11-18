"""Closed-form round-budget estimators from the manuscript formulas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence

import numpy as np

from boosting_qml.losses.margins import empirical_margin_loss, normalized_margin
from boosting_qml.types import BoostingRoundResult, RoundBudgetBoundFamily

_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class RoundScalingPoint:
    eta: float
    margin_threshold: float
    target_margin_loss: float
    round_budget_real_adaboost: float
    round_budget_adaboost_r: float


@dataclass(frozen=True)
class RunningRoundBudgetEstimate:
    round_index: int
    edge_t: float
    edge_lower_bound_t: float
    alpha_l1_norm_t: float
    estimated_round_budget_t: float


@dataclass(frozen=True)
class RunningConvergenceDiagnostics:
    round_index: int
    edge_t: float
    edge_pathwise_min_t: float
    edge_lcb_t: float
    edge_lcb_pathwise_min_t: float
    edge_conf_radius_t: float
    edge_conf_delta_t: float
    edge_effective_sample_size_t: float
    cumulative_edge_sq_t: float
    avg_edge_sq_t: float
    edge_effective_t: float
    edge_percentile_t: float
    edge_cvar_t: float
    alpha_l1_norm_t: float
    exp_potential_t: float
    log_exp_potential_t: float
    exp_potential_ratio_t: float
    logistic_potential_t: float
    normalized_margin_min_t: float
    normalized_margin_q05_t: float
    normalized_margin_q10_t: float
    normalized_margin_q25_t: float
    normalized_margin_q50_t: float
    normalized_margin_q75_t: float
    normalized_margin_q90_t: float
    normalized_margin_max_t: float
    equilibrium_margin_t: float
    estimated_round_budget_pathwise_t: float
    estimated_round_budget_lcb_t: float
    estimated_round_budget_effective_t: float
    estimated_round_budget_percentile_t: float
    estimated_round_budget_cvar_t: float
    surrogate_edge_lower_bound_t: float = math.nan
    surrogate_edge_pathwise_min_t: float = math.nan
    estimated_round_budget_surrogate_pathwise_t: float = math.inf


def _format_round_budget_value(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    if value <= 0.0:
        return f"{value:.3g}"
    if value >= 1.0e7:
        return f"{value:.3e}"
    return str(int(math.ceil(value)))


def _format_edge_value(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.4g}"


def format_running_convergence_diagnostics(row: RunningConvergenceDiagnostics) -> str:
    """Compact log string for the multi-bound round-budget diagnostics.

    The analytic estimators return real-valued budgets. Logs show their ceiling
    because a run can only execute an integer number of boosting rounds. The
    full floating-point values remain available in ``RunningConvergenceDiagnostics``
    for plotting and offline analysis.
    """

    return (
        "T*[path={path},lcb={lcb},eff={eff},pctl={pctl},cvar={cvar},surr={surr}] "
        "eta[path={eta_path},lcb={eta_lcb},eff={eta_eff},pctl={eta_pctl},"
        "cvar={eta_cvar},surr={eta_surr}]"
    ).format(
        path=_format_round_budget_value(row.estimated_round_budget_pathwise_t),
        lcb=_format_round_budget_value(row.estimated_round_budget_lcb_t),
        eff=_format_round_budget_value(row.estimated_round_budget_effective_t),
        pctl=_format_round_budget_value(row.estimated_round_budget_percentile_t),
        cvar=_format_round_budget_value(row.estimated_round_budget_cvar_t),
        surr=_format_round_budget_value(row.estimated_round_budget_surrogate_pathwise_t),
        eta_path=_format_edge_value(row.edge_pathwise_min_t),
        eta_lcb=_format_edge_value(row.edge_lcb_pathwise_min_t),
        eta_eff=_format_edge_value(row.edge_effective_t),
        eta_pctl=_format_edge_value(row.edge_percentile_t),
        eta_cvar=_format_edge_value(row.edge_cvar_t),
        eta_surr=_format_edge_value(row.surrogate_edge_pathwise_min_t),
    )


def _validate_eta(eta: float) -> None:
    if not (0.0 < eta < 1.0):
        raise ValueError("eta must lie in (0, 1).")


def _validate_margin_threshold(margin_threshold: float) -> None:
    if not (0.0 < margin_threshold < 1.0):
        raise ValueError("margin_threshold must lie in (0, 1).")


def _validate_target(target_margin_loss: float) -> None:
    if not (0.0 < target_margin_loss < 1.0):
        raise ValueError("target_margin_loss must lie in (0, 1).")


def _validate_quantile(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must lie in (0, 1).")


def _validate_tail_fraction(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must lie in (0, 1].")


def real_adaboost_contraction_base(eta: float, margin_threshold: float) -> float:
    """B_std(mu, eta) from Eq. (margin_bound_real_ab_uniform)."""

    _validate_eta(eta)
    _validate_margin_threshold(margin_threshold)
    return float(
        np.sqrt(
            ((1.0 - eta) ** (1.0 - margin_threshold))
            * ((1.0 + eta) ** (1.0 + margin_threshold))
        )
    )


def estimate_round_budget_real_adaboost(
    eta: float,
    margin_threshold: float,
    target_margin_loss: float,
) -> float:
    """T*_std from Eq. (Tstar_real_adaboost). Returns inf if bound is vacuous."""

    _validate_eta(eta)
    _validate_margin_threshold(margin_threshold)
    _validate_target(target_margin_loss)

    base = real_adaboost_contraction_base(eta, margin_threshold)
    if not np.isfinite(base) or base >= 1.0:
        return math.inf

    return float(np.log(1.0 / target_margin_loss) / (-np.log(base + _EPS)))


def _adaboost_r_multiplicative_factor(tau: float) -> float:
    tau = float(np.clip(tau, -1.0 + 1e-12, 1.0 - 1e-12))
    return float(max(1.0, (1.0 + tau) / (1.0 - tau)))


def estimate_round_budget_adaboost_r(
    eta: float,
    margin_threshold: float,
    target_margin_loss: float,
    alpha_l1_norm: float = 1.0,
) -> float:
    """T*_R from Eq. (Tstar_adaboostr_mu_final)."""

    _validate_eta(eta)
    _validate_margin_threshold(margin_threshold)
    _validate_target(target_margin_loss)
    if alpha_l1_norm <= 0.0:
        raise ValueError("alpha_l1_norm must be positive.")

    tau_mu = float(np.tanh((margin_threshold * alpha_l1_norm) / 2.0))
    m_tau = _adaboost_r_multiplicative_factor(tau_mu)

    return float((2.0 / (eta ** 2)) * (np.log(m_tau + _EPS) + np.log(1.0 / target_margin_loss)))


def compare_round_scaling_grid(
    eta_values: Iterable[float],
    margin_values: Iterable[float],
    target_margin_loss: float,
    alpha_l1_norm: float = 1.0,
) -> List[RoundScalingPoint]:
    """Compute side-by-side round budgets over an eta x margin grid."""

    _validate_target(target_margin_loss)
    points: List[RoundScalingPoint] = []
    for eta in eta_values:
        for margin in margin_values:
            t_real = estimate_round_budget_real_adaboost(
                eta=eta,
                margin_threshold=margin,
                target_margin_loss=target_margin_loss,
            )
            t_r = estimate_round_budget_adaboost_r(
                eta=eta,
                margin_threshold=margin,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_norm,
            )
            points.append(
                RoundScalingPoint(
                    eta=float(eta),
                    margin_threshold=float(margin),
                    target_margin_loss=float(target_margin_loss),
                    round_budget_real_adaboost=t_real,
                    round_budget_adaboost_r=t_r,
                )
            )
    return points


def _validate_confidence_delta(confidence_delta: float) -> None:
    if not (0.0 < confidence_delta < 1.0):
        raise ValueError("confidence_delta must lie in (0, 1).")


def _time_uniform_delta_t(confidence_delta: float, round_index: int) -> float:
    # Union bound with sum_t 1/t^2 = pi^2/6.
    return float(max(_EPS, confidence_delta * (6.0 / (math.pi ** 2)) / (round_index ** 2)))


def _scores_for_edge_statistics(round_result: BoostingRoundResult) -> np.ndarray:
    scores = np.asarray(round_result.train_scores, dtype=np.float64)
    if round_result.h_t_star is not None and round_result.h_t_star > _EPS:
        scores = scores / float(round_result.h_t_star)
    return np.clip(scores, -1.0, 1.0)


def _edge_lcb_statistics(
    round_result: BoostingRoundResult,
    y_train: np.ndarray,
    confidence_delta_t: float,
) -> tuple[float, float, float, float]:
    weights = np.asarray(round_result.example_weights_before_update, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("example_weights_before_update must be a rank-1 array.")
    if y_train.shape[0] != weights.shape[0]:
        raise ValueError("y_train and example_weights_before_update must have the same length.")

    scores_for_edge = _scores_for_edge_statistics(round_result)
    if scores_for_edge.shape[0] != weights.shape[0]:
        raise ValueError("train_scores and example_weights_before_update must have the same length.")

    weighted_labels_scores = y_train * scores_for_edge
    edge_mean = float(np.dot(weights, weighted_labels_scores))
    centered = weighted_labels_scores - edge_mean
    weighted_var = float(np.dot(weights, centered * centered))
    n_eff = float(1.0 / max(np.sum(weights * weights), _EPS))

    log_term = float(max(0.0, np.log(3.0 / max(confidence_delta_t, _EPS))))
    radius = float(np.sqrt(max(0.0, 2.0 * weighted_var * log_term / max(n_eff, 1.0))))
    radius += float((3.0 * log_term) / max(n_eff, 1.0))
    edge_lcb = edge_mean - radius
    return edge_mean, edge_lcb, radius, n_eff


def _safe_estimated_round_budget_real(
    eta: float,
    margin_threshold: float,
    target_margin_loss: float,
) -> float:
    if not np.isfinite(eta) or eta <= 0.0:
        return math.inf
    eta_clipped = min(max(float(eta), 1e-12), 1.0 - 1e-12)
    return estimate_round_budget_real_adaboost(
        eta=eta_clipped,
        margin_threshold=margin_threshold,
        target_margin_loss=target_margin_loss,
    )


def _safe_estimated_round_budget_adaboost_r(
    eta: float,
    margin_threshold: float,
    target_margin_loss: float,
    alpha_l1_norm: float,
) -> float:
    if not np.isfinite(eta) or eta <= 0.0:
        return math.inf
    eta_clipped = min(max(float(eta), 1e-12), 1.0 - 1e-12)
    alpha_l1 = max(float(alpha_l1_norm), 1e-12)
    return estimate_round_budget_adaboost_r(
        eta=eta_clipped,
        margin_threshold=margin_threshold,
        target_margin_loss=target_margin_loss,
        alpha_l1_norm=alpha_l1,
    )


def _lower_tail_cvar(values: np.ndarray, tail_fraction: float) -> float:
    if values.size == 0:
        raise ValueError("values must be non-empty.")
    n_tail = max(1, int(math.ceil(float(tail_fraction) * values.size)))
    sorted_values = np.sort(values)
    return float(np.mean(sorted_values[:n_tail]))


def estimate_round_budget_from_cumulative_edge(
    cumulative_edge_sq: float,
    n_rounds: int,
    target_margin_loss: float,
) -> float:
    """
    Estimate T* from cumulative edge energy via average gamma^2.

    Uses the classical shape exp(-2 * sum_t r_t^2), yielding:
    T* ~= log(1/delta) / (2 * mean_t r_t^2).
    """

    _validate_target(target_margin_loss)
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive.")
    avg_edge_sq = float(cumulative_edge_sq / n_rounds)
    if not np.isfinite(avg_edge_sq) or avg_edge_sq <= 0.0:
        return math.inf
    return float(np.log(1.0 / target_margin_loss) / (2.0 * avg_edge_sq))


def estimate_equilibrium_margin(
    margins: np.ndarray,
    grid_size: int = 257,
) -> float:
    """
    Estimate an equilibrium margin proxy theta where margin_loss(theta) ~ theta.

    If feasible points exist with margin_loss(theta) <= theta, returns the largest
    such theta. Otherwise returns the closest fixed-point approximation.
    """

    if grid_size < 16:
        raise ValueError("grid_size must be at least 16.")

    clipped_margins = np.clip(np.asarray(margins, dtype=np.float64), -1.0, 1.0)
    positive = clipped_margins[clipped_margins > 0.0]
    if positive.size == 0:
        return 0.0

    grid = np.linspace(1e-6, 1.0, grid_size)
    quantiles = np.quantile(positive, q=np.linspace(0.0, 1.0, 33))
    candidates = np.unique(np.clip(np.concatenate([grid, quantiles]), 1e-6, 1.0))

    best_feasible = 0.0
    best_theta = float(candidates[0])
    best_gap = math.inf
    for theta in candidates:
        theta_value = float(theta)
        loss_value = empirical_margin_loss(clipped_margins, theta_value)
        gap = abs(loss_value - theta_value)
        if gap < best_gap:
            best_gap = gap
            best_theta = theta_value
        if loss_value <= theta_value and theta_value > best_feasible:
            best_feasible = theta_value

    return float(best_feasible if best_feasible > 0.0 else best_theta)


def estimate_running_convergence_diagnostics(
    round_results: Sequence[BoostingRoundResult],
    y_train: np.ndarray,
    margin_threshold: float,
    target_margin_loss: float,
    algorithm: str,
    confidence_delta: float = 0.05,
    equilibrium_margin_grid_size: int = 257,
    lower_percentile_q: float = 0.10,
    lower_cvar_tail_fraction: float = 0.20,
) -> List[RunningConvergenceDiagnostics]:
    """
    Compute per-round convergence diagnostics for Real AdaBoost or AdaBoost.R.

    Includes:
    - pathwise and LCB edge lower bounds,
    - cumulative edge energy / effective edge,
    - exponential and logistic potentials,
    - normalized margin distribution + equilibrium margin,
    - T* estimates from pathwise, LCB, effective-edge, percentile, and CVaR variants.
    """

    _validate_margin_threshold(margin_threshold)
    _validate_target(target_margin_loss)
    _validate_confidence_delta(confidence_delta)
    _validate_quantile("lower_percentile_q", lower_percentile_q)
    _validate_tail_fraction("lower_cvar_tail_fraction", lower_cvar_tail_fraction)
    if algorithm not in {"real_adaboost", "adaboost_r"}:
        raise ValueError("algorithm must be one of {'real_adaboost', 'adaboost_r'}.")

    y_arr = np.asarray(y_train, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y_train must be a rank-1 array.")
    if not round_results:
        return []
    if len(round_results[0].train_scores) != y_arr.shape[0]:
        raise ValueError("y_train length must match round_result.train_scores length.")

    rows: List[RunningConvergenceDiagnostics] = []
    running_edge_min = math.inf
    running_edge_lcb_min = math.inf
    running_alpha_l1 = 0.0
    running_surrogate_edge_min = math.inf
    cumulative_edge_sq = 0.0
    previous_exp_potential = 1.0
    edge_history: List[float] = []

    for idx, result in enumerate(round_results, start=1):
        edge_t = float(result.edge_t)
        running_edge_min = min(running_edge_min, edge_t)
        running_alpha_l1 += abs(float(result.boost_coef_t))
        alpha_l1_t = max(running_alpha_l1, 1e-12)

        confidence_delta_t = _time_uniform_delta_t(confidence_delta, idx)
        edge_empirical_t, edge_lcb_t, edge_radius_t, n_eff_t = _edge_lcb_statistics(
            round_result=result,
            y_train=y_arr,
            confidence_delta_t=confidence_delta_t,
        )
        # Prefer the explicitly logged edge_t to avoid tiny numerical mismatch.
        edge_t_used = edge_t if np.isfinite(edge_t) else edge_empirical_t
        running_edge_lcb_min = min(running_edge_lcb_min, edge_lcb_t)

        cumulative_edge_sq += edge_t_used * edge_t_used
        avg_edge_sq_t = float(cumulative_edge_sq / idx)
        edge_effective_t = float(np.sqrt(max(0.0, avg_edge_sq_t)))

        surrogate_edge_lb_t = math.nan
        surrogate_edge_pathwise_min_t = math.nan
        if result.surrogate_edge_lower_bound_t is not None:
            surrogate_candidate = float(result.surrogate_edge_lower_bound_t)
            if np.isfinite(surrogate_candidate):
                surrogate_edge_lb_t = surrogate_candidate
                running_surrogate_edge_min = min(running_surrogate_edge_min, surrogate_candidate)
                surrogate_edge_pathwise_min_t = running_surrogate_edge_min

        edge_history.append(edge_t_used)
        edge_hist_arr = np.asarray(edge_history, dtype=np.float64)
        edge_percentile_t = float(np.quantile(edge_hist_arr, q=lower_percentile_q))
        edge_cvar_t = _lower_tail_cvar(edge_hist_arr, tail_fraction=lower_cvar_tail_fraction)

        raw_margin_t = y_arr * np.asarray(result.train_ensemble_score, dtype=np.float64)
        exp_potential_t = float(np.mean(np.exp(np.clip(-raw_margin_t, -60.0, 60.0))))
        log_exp_potential_t = float(np.log(max(exp_potential_t, _EPS)))
        exp_potential_ratio_t = float(exp_potential_t / max(previous_exp_potential, _EPS))
        previous_exp_potential = exp_potential_t
        logistic_potential_t = float(np.mean(np.logaddexp(0.0, -raw_margin_t)))

        normalized_margin_t = normalized_margin(
            y_arr,
            np.asarray(result.train_ensemble_score, dtype=np.float64),
            alpha_l1_norm=alpha_l1_t,
        )
        equilibrium_margin_t = estimate_equilibrium_margin(
            normalized_margin_t,
            grid_size=equilibrium_margin_grid_size,
        )

        if algorithm == "real_adaboost":
            t_path = _safe_estimated_round_budget_real(
                eta=running_edge_min,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
            t_lcb = _safe_estimated_round_budget_real(
                eta=running_edge_lcb_min,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
            t_eff = _safe_estimated_round_budget_real(
                eta=edge_effective_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
            t_percentile = _safe_estimated_round_budget_real(
                eta=edge_percentile_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
            t_cvar = _safe_estimated_round_budget_real(
                eta=edge_cvar_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
            t_surrogate = _safe_estimated_round_budget_real(
                eta=surrogate_edge_pathwise_min_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
            )
        else:
            t_surrogate = _safe_estimated_round_budget_adaboost_r(
                eta=surrogate_edge_pathwise_min_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )
            t_path = _safe_estimated_round_budget_adaboost_r(
                eta=running_edge_min,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )
            t_lcb = _safe_estimated_round_budget_adaboost_r(
                eta=running_edge_lcb_min,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )
            t_eff = _safe_estimated_round_budget_adaboost_r(
                eta=edge_effective_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )
            t_percentile = _safe_estimated_round_budget_adaboost_r(
                eta=edge_percentile_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )
            t_cvar = _safe_estimated_round_budget_adaboost_r(
                eta=edge_cvar_t,
                margin_threshold=margin_threshold,
                target_margin_loss=target_margin_loss,
                alpha_l1_norm=alpha_l1_t,
            )

        quantiles = np.quantile(normalized_margin_t, q=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append(
            RunningConvergenceDiagnostics(
                round_index=idx,
                edge_t=edge_t_used,
                edge_pathwise_min_t=running_edge_min,
                edge_lcb_t=edge_lcb_t,
                edge_lcb_pathwise_min_t=running_edge_lcb_min,
                edge_conf_radius_t=edge_radius_t,
                edge_conf_delta_t=confidence_delta_t,
                edge_effective_sample_size_t=n_eff_t,
                cumulative_edge_sq_t=cumulative_edge_sq,
                avg_edge_sq_t=avg_edge_sq_t,
                edge_effective_t=edge_effective_t,
                edge_percentile_t=edge_percentile_t,
                edge_cvar_t=edge_cvar_t,
                alpha_l1_norm_t=alpha_l1_t,
                exp_potential_t=exp_potential_t,
                log_exp_potential_t=log_exp_potential_t,
                exp_potential_ratio_t=exp_potential_ratio_t,
                logistic_potential_t=logistic_potential_t,
                normalized_margin_min_t=float(np.min(normalized_margin_t)),
                normalized_margin_q05_t=float(quantiles[0]),
                normalized_margin_q10_t=float(quantiles[1]),
                normalized_margin_q25_t=float(quantiles[2]),
                normalized_margin_q50_t=float(quantiles[3]),
                normalized_margin_q75_t=float(quantiles[4]),
                normalized_margin_q90_t=float(quantiles[5]),
                normalized_margin_max_t=float(np.max(normalized_margin_t)),
                equilibrium_margin_t=equilibrium_margin_t,
                estimated_round_budget_pathwise_t=t_path,
                estimated_round_budget_lcb_t=t_lcb,
                estimated_round_budget_effective_t=t_eff,
                estimated_round_budget_percentile_t=t_percentile,
                estimated_round_budget_cvar_t=t_cvar,
                surrogate_edge_lower_bound_t=surrogate_edge_lb_t,
                surrogate_edge_pathwise_min_t=surrogate_edge_pathwise_min_t,
                estimated_round_budget_surrogate_pathwise_t=t_surrogate,
            )
        )
    return rows


def extract_round_budget_bound_families(
    diagnostics_rows: Sequence[RunningConvergenceDiagnostics],
) -> List[RoundBudgetBoundFamily]:
    """Project full diagnostics rows onto a compact multi-bound T schema."""

    return [
        RoundBudgetBoundFamily(
            round_index=row.round_index,
            edge_pathwise_min_t=row.edge_pathwise_min_t,
            edge_lcb_pathwise_min_t=row.edge_lcb_pathwise_min_t,
            edge_effective_t=row.edge_effective_t,
            edge_percentile_t=row.edge_percentile_t,
            edge_cvar_t=row.edge_cvar_t,
            estimated_round_budget_pathwise_t=row.estimated_round_budget_pathwise_t,
            estimated_round_budget_lcb_t=row.estimated_round_budget_lcb_t,
            estimated_round_budget_effective_t=row.estimated_round_budget_effective_t,
            estimated_round_budget_percentile_t=row.estimated_round_budget_percentile_t,
            estimated_round_budget_cvar_t=row.estimated_round_budget_cvar_t,
            surrogate_edge_lower_bound_t=(
                None if not np.isfinite(row.surrogate_edge_lower_bound_t) else row.surrogate_edge_lower_bound_t
            ),
            surrogate_edge_pathwise_min_t=(
                None if not np.isfinite(row.surrogate_edge_pathwise_min_t) else row.surrogate_edge_pathwise_min_t
            ),
            estimated_round_budget_surrogate_pathwise_t=(
                None
                if not np.isfinite(row.estimated_round_budget_surrogate_pathwise_t)
                else row.estimated_round_budget_surrogate_pathwise_t
            ),
        )
        for row in diagnostics_rows
    ]


def estimate_running_round_budget_real_adaboost(
    edges: Iterable[float],
    margin_threshold: float,
    target_margin_loss: float,
    edge_floor: float = 1e-6,
) -> List[RunningRoundBudgetEstimate]:
    """
    Re-estimate Real AdaBoost T* at each round from the running edge lower bound.

    edge_lower_bound_t = min_{s<=t} edge_s.
    """

    _validate_margin_threshold(margin_threshold)
    _validate_target(target_margin_loss)
    if edge_floor <= 0.0:
        raise ValueError("edge_floor must be positive.")

    running_min = math.inf
    rows: List[RunningRoundBudgetEstimate] = []
    for idx, edge_t in enumerate(edges, start=1):
        edge_value = float(edge_t)
        running_min = min(running_min, edge_value)
        eta_t = max(running_min, edge_floor)
        t_star_t = estimate_round_budget_real_adaboost(
            eta=eta_t,
            margin_threshold=margin_threshold,
            target_margin_loss=target_margin_loss,
        )
        rows.append(
            RunningRoundBudgetEstimate(
                round_index=idx,
                edge_t=edge_value,
                edge_lower_bound_t=running_min,
                alpha_l1_norm_t=0.0,
                estimated_round_budget_t=t_star_t,
            )
        )
    return rows


def estimate_running_round_budget_adaboost_r(
    edges: Iterable[float],
    boost_coefs: Iterable[float],
    margin_threshold: float,
    target_margin_loss: float,
    edge_floor: float = 1e-6,
) -> List[RunningRoundBudgetEstimate]:
    """
    Re-estimate AdaBoost.R T* at each round from running edge lower bound and ||alpha||_1.

    edge_lower_bound_t = min_{s<=t} edge_s
    alpha_l1_norm_t = sum_{s<=t} |alpha_s|
    """

    _validate_margin_threshold(margin_threshold)
    _validate_target(target_margin_loss)
    if edge_floor <= 0.0:
        raise ValueError("edge_floor must be positive.")

    edge_list = [float(v) for v in edges]
    coef_list = [float(v) for v in boost_coefs]
    if len(edge_list) != len(coef_list):
        raise ValueError("edges and boost_coefs must have the same length.")

    running_min = math.inf
    running_alpha_l1 = 0.0
    rows: List[RunningRoundBudgetEstimate] = []
    for idx, (edge_t, coef_t) in enumerate(zip(edge_list, coef_list), start=1):
        running_min = min(running_min, edge_t)
        running_alpha_l1 += abs(coef_t)
        eta_t = max(running_min, edge_floor)
        alpha_l1_t = max(running_alpha_l1, edge_floor)
        t_star_t = estimate_round_budget_adaboost_r(
            eta=eta_t,
            margin_threshold=margin_threshold,
            target_margin_loss=target_margin_loss,
            alpha_l1_norm=alpha_l1_t,
        )
        rows.append(
            RunningRoundBudgetEstimate(
                round_index=idx,
                edge_t=edge_t,
                edge_lower_bound_t=running_min,
                alpha_l1_norm_t=alpha_l1_t,
                estimated_round_budget_t=t_star_t,
            )
        )
    return rows
