"""Linear-programming utilities for robust weak-learner banks.

For a matrix ``A[k, i] = y_i g_k(x_i)``, the robust bank edge is the
zero-sum-game value

    min_D max_k (A D)_k = max_lambda min_i (A.T lambda)_i.

The routines here are deterministic postprocessing only; they never evaluate
or train a weak learner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, linprog

from boosting_qml.analysis.theory_alignment import real_margin_contraction


@dataclass(frozen=True)
class RobustBankSolution:
    """One primal or dual solution of the robust-bank game."""

    value: float
    formulation: str
    learner_mixture: np.ndarray | None = None
    adversarial_distribution: np.ndarray | None = None


def _advantages(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("advantages must have shape [K, m] with K,m > 0")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("advantages must be finite")
    if np.any(np.abs(matrix) > 1.0 + 1e-10):
        raise ValueError("advantages must lie in [-1, 1]")
    return np.clip(matrix, -1.0, 1.0)


def solve_robust_bank_dual(advantages: np.ndarray) -> RobustBankSolution:
    """Solve ``max_lambda min_i sum_k lambda_k A[k,i]``."""

    matrix = _advantages(advantages)
    k, m = matrix.shape
    # Variables are (lambda_1,...,lambda_K, eta).  A.T lambda >= eta.
    result = linprog(
        np.concatenate((np.zeros(k), [-1.0])),
        A_ub=np.column_stack((-matrix.T, np.ones(m))),
        b_ub=np.zeros(m),
        A_eq=np.concatenate((np.ones(k), [0.0]))[None, :],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * k + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"robust-bank dual LP failed: {result.message}")
    mixture = np.asarray(result.x[:k], dtype=np.float64)
    value = float(np.min(matrix.T @ mixture))
    return RobustBankSolution(value, "dual", learner_mixture=mixture)


def solve_robust_bank_primal(advantages: np.ndarray) -> RobustBankSolution:
    """Solve ``min_D max_k sum_i D_i A[k,i]``."""

    matrix = _advantages(advantages)
    k, m = matrix.shape
    # Variables are (D_1,...,D_m, eta).  A D <= eta.
    result = linprog(
        np.concatenate((np.zeros(m), [1.0])),
        A_ub=np.column_stack((matrix, -np.ones(k))),
        b_ub=np.zeros(k),
        A_eq=np.concatenate((np.ones(m), [0.0]))[None, :],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * m + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"robust-bank primal LP failed: {result.message}")
    distribution = np.asarray(result.x[:m], dtype=np.float64)
    value = float(np.max(matrix @ distribution))
    return RobustBankSolution(value, "primal", adversarial_distribution=distribution)


def robust_bank_edge(advantages: np.ndarray) -> float:
    """Return the dual robust-bank value."""

    return solve_robust_bank_dual(advantages).value


def bank_lcb(estimated_eta: float, error_radius: float) -> float:
    """Translate a simultaneous score band to a robust-edge lower bound."""

    eta = float(estimated_eta)
    radius = float(error_radius)
    if not math.isfinite(eta) or not math.isfinite(radius) or radius < 0.0:
        raise ValueError("eta and a nonnegative error radius must be finite")
    return eta - radius


def shots_for_simultaneous_radius(
    *, error_radius: float, family_size: int, confidence_delta: float
) -> int | None:
    """Minimum Hoeffding shot count making the radius strictly smaller.

    Returns ``None`` when the requested radius is nonpositive.  The strict
    inequality is useful for positive-LCB and strict-crossover certificates.
    """

    radius = float(error_radius)
    family = int(family_size)
    delta = float(confidence_delta)
    if family <= 0 or not (0.0 < delta < 1.0):
        raise ValueError("family_size must be positive and delta in (0,1)")
    if radius <= 0.0 or not math.isfinite(radius):
        return None
    raw = 2.0 * math.log(2.0 * family / delta) / (radius * radius)
    return int(math.floor(raw) + 1)


def real_contraction_edge_threshold(mu: float) -> float:
    """Smallest positive edge above which ``B_real(mu, eta) < 1``."""

    mu = float(mu)
    if not (0.0 < mu < 1.0):
        raise ValueError("mu must lie in (0,1)")
    return float(brentq(lambda eta: math.log(real_margin_contraction(mu, eta)), mu, 1.0 - 1e-12))


def real_crossover_edge_threshold(
    *, mu: float, epsilon_mar: float, gamma_squared: float
) -> float | None:
    """Edge needed for the strict integer rule ``T_bank < gamma^2``.

    The largest permitted integer round count is ``ceil(gamma^2)-1``.
    """

    epsilon = float(epsilon_mar)
    gamma2 = float(gamma_squared)
    if not (0.0 < epsilon < 1.0) or gamma2 <= 0.0:
        raise ValueError("epsilon must lie in (0,1) and gamma_squared be positive")
    max_rounds = int(math.ceil(gamma2) - 1)
    if max_rounds < 1:
        return None
    target_log_b = math.log(epsilon) / max_rounds
    contraction_edge = real_contraction_edge_threshold(mu)
    objective = lambda eta: math.log(real_margin_contraction(mu, eta)) - target_log_b
    if objective(1.0 - 1e-12) > 0.0:
        return None
    return float(brentq(objective, contraction_edge, 1.0 - 1e-12))


def verify_bank_guarantee(
    advantages: np.ndarray,
    distributions: np.ndarray,
    eta_bank: float,
) -> np.ndarray:
    """Return ``max_k edge_D(g_k)-eta_bank`` for supplied distributions."""

    matrix = _advantages(advantages)
    weights = np.asarray(distributions, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != matrix.shape[1]:
        raise ValueError("distributions must have shape [n, m]")
    if np.any(weights < -1e-12) or not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-9):
        raise ValueError("each distribution must be nonnegative and sum to one")
    return np.max(matrix @ weights.T, axis=0) - float(eta_bank)


__all__ = [
    "RobustBankSolution",
    "bank_lcb",
    "real_contraction_edge_threshold",
    "real_crossover_edge_threshold",
    "robust_bank_edge",
    "shots_for_simultaneous_radius",
    "solve_robust_bank_dual",
    "solve_robust_bank_primal",
    "verify_bank_guarantee",
]
