"""Deterministic mathematics for exponential majorants of ramp loss."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def ramp_phi(z: np.ndarray | float, mu: float) -> np.ndarray | float:
    """The paper's piecewise-linear empirical ramp loss."""

    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    values = np.asarray(z, dtype=np.float64)
    result = np.where(
        values <= 0.0,
        1.0,
        np.where(values < mu, 1.0 - values / mu, 0.0),
    )
    return float(result) if result.ndim == 0 else result


def log_majorant_constant(lam: float, mu: float) -> float:
    """Return log C(lambda, mu) for the tight exponential majorant."""

    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda must be finite and positive")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    a = lam * mu
    return 0.0 if a <= 1.0 else a - 1.0 - math.log(a)


def majorant_constant(lam: float, mu: float) -> float:
    return math.exp(log_majorant_constant(lam, mu))


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("logmeanexp requires at least one value")
    peak = float(np.max(values))
    return peak + math.log(float(np.mean(np.exp(values - peak))))


def log_exponential_ramp_bound(lam: float, margins: np.ndarray, mu: float) -> float:
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1 or margins.size == 0 or not np.all(np.isfinite(margins)):
        raise ValueError("margins must be a nonempty finite vector")
    return log_majorant_constant(lam, mu) + logmeanexp(-lam * margins)


def _objective_derivative(lam: float, margins: np.ndarray, mu: float) -> float:
    exponents = -lam * margins
    exponents -= float(np.max(exponents))
    weights = np.exp(exponents)
    tilted_mean = float(np.dot(weights, margins) / np.sum(weights))
    c_derivative = 0.0 if lam * mu <= 1.0 else mu - 1.0 / lam
    return c_derivative - tilted_mean


@dataclass(frozen=True)
class OptimizedRampBound:
    lambda_opt: float
    log_bound: float
    status: str
    bracket_low: float | None
    bracket_high: float | None
    derivative_low: float | None
    derivative_high: float | None
    grid_min_log_bound: float
    grid_gap_from_optimum: float
    iterations: int


def optimize_exponential_ramp_bound(
    margins: np.ndarray,
    mu: float,
    *,
    derivative_tolerance: float = 2e-13,
    max_iterations: int = 90,
) -> OptimizedRampBound:
    """Globally minimize the one-temperature ramp bound in log space.

    ``log C`` and ``logmeanexp(-lambda*z)`` are convex.  Their derivative is
    monotone, so a sign-changing derivative bracket and bisection give the
    global minimizer.  The two non-attained boundary infima are handled
    analytically: lambda -> 0+ gives one, while lambda -> infinity gives zero
    when every margin is at least ``mu``.
    """

    z = np.asarray(margins, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("margins must be a nonempty finite vector")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")

    if float(np.min(z)) >= mu:
        return OptimizedRampBound(
            lambda_opt=math.inf,
            log_bound=-math.inf,
            status="infimum_at_lambda_infinity_all_margins_at_least_mu",
            bracket_low=None,
            bracket_high=None,
            derivative_low=None,
            derivative_high=None,
            grid_min_log_bound=-math.inf,
            grid_gap_from_optimum=0.0,
            iterations=0,
        )

    derivative_zero = -float(np.mean(z))
    if derivative_zero >= 0.0:
        return OptimizedRampBound(
            lambda_opt=0.0,
            log_bound=0.0,
            status="infimum_at_lambda_zero",
            bracket_low=0.0,
            bracket_high=1.0 / mu,
            derivative_low=derivative_zero,
            derivative_high=_objective_derivative(1.0 / mu, z, mu),
            grid_min_log_bound=0.0,
            grid_gap_from_optimum=0.0,
            iterations=0,
        )

    low = 0.0
    high = max(1.0, 1.0 / mu)
    derivative_high = _objective_derivative(high, z, mu)
    expansion_count = 0
    while derivative_high <= 0.0:
        high *= 2.0
        expansion_count += 1
        if not math.isfinite(high) or expansion_count > 100:
            raise ArithmeticError("failed to bracket finite convex optimum")
        derivative_high = _objective_derivative(high, z, mu)

    derivative_low = derivative_zero
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        middle = 0.5 * (low + high)
        derivative_middle = _objective_derivative(middle, z, mu)
        if derivative_middle <= 0.0:
            low = middle
            derivative_low = derivative_middle
        else:
            high = middle
            derivative_high = derivative_middle
        if high - low <= derivative_tolerance * max(1.0, middle):
            break

    lam = 0.5 * (low + high)
    log_bound = log_exponential_ramp_bound(lam, z, mu)

    # Independent, deliberately broad multiplicative grid.  It excludes the
    # bisection result itself, so its minimum is a genuine cross-check.
    candidates = {
        max(np.finfo(float).tiny, lam * factor)
        for factor in (1.0 / 256.0, 1.0 / 64.0, 1.0 / 16.0, 0.25, 0.5, 2.0, 4.0, 16.0, 64.0, 256.0)
    }
    candidates.update((1.0 / mu, max(np.finfo(float).tiny, low), high))
    grid_min = min(log_exponential_ramp_bound(value, z, mu) for value in candidates)
    return OptimizedRampBound(
        lambda_opt=lam,
        log_bound=log_bound,
        status="finite_convex_minimum",
        bracket_low=low,
        bracket_high=high,
        derivative_low=derivative_low,
        derivative_high=derivative_high,
        grid_min_log_bound=grid_min,
        grid_gap_from_optimum=grid_min - log_bound,
        iterations=iterations,
    )
