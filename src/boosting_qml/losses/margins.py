"""Margin-related metrics used by boosting algorithms."""

from __future__ import annotations

import numpy as np

from boosting_qml.types import Array

_EPS = np.finfo(np.float64).eps


def raw_margin(y_true: Array, ensemble_score: Array) -> Array:
    y = np.asarray(y_true, dtype=np.float64)
    f = np.asarray(ensemble_score, dtype=np.float64)
    return y * f


def normalized_margin(y_true: Array, ensemble_score: Array, alpha_l1_norm: float) -> Array:
    if alpha_l1_norm <= 0.0:
        return np.zeros_like(ensemble_score, dtype=np.float64)
    return raw_margin(y_true, ensemble_score) / (alpha_l1_norm + _EPS)


def empirical_margin_loss(margins: Array, margin_threshold: float) -> float:
    if margin_threshold <= 0.0:
        raise ValueError("margin_threshold must be positive.")

    margins = np.asarray(margins, dtype=np.float64)
    losses = np.where(
        margins >= margin_threshold,
        0.0,
        np.where(margins >= 0.0, 1.0 - margins / margin_threshold, 1.0),
    )
    return float(np.mean(losses))


def ramp_loss(margins: Array, mu: float) -> Array:
    """Evaluate the pointwise ramp loss ``Phi_mu``.

    The endpoint convention is explicit: ``Phi_mu(0) = 1`` and
    ``Phi_mu(mu) = 0``.  The returned array has the same shape as ``margins``.
    """

    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive.")
    values = np.asarray(margins, dtype=np.float64)
    return np.where(
        values <= 0.0,
        1.0,
        np.where(values < float(mu), 1.0 - values / float(mu), 0.0),
    )


def empirical_ramp_loss(margins: Array, mu: float) -> float:
    """Return the empirical mean of :func:`ramp_loss`."""

    values = ramp_loss(margins, mu)
    if values.size == 0:
        raise ValueError("margins must contain at least one value.")
    return float(np.mean(values))


def normalized_ensemble_score(ensemble_score: Array, alpha_l1_norm: float) -> Array:
    """Normalize an ensemble score by a strictly positive coefficient mass."""

    if not np.isfinite(alpha_l1_norm) or alpha_l1_norm <= 0.0:
        raise ValueError("alpha_l1_norm must be finite and positive.")
    return np.asarray(ensemble_score, dtype=np.float64) / float(alpha_l1_norm)


def normalized_empirical_ramp_loss(
    y_true: Array,
    ensemble_score: Array,
    alpha_l1_norm: float,
    mu: float,
) -> float:
    """Empirical ramp loss of the normalized ensemble ``F_T / ||alpha||_1``.

    This is the canonical online/offline definition used for frozen
    theory-aligned margin targets.  Keeping the normalization and ramp
    evaluation together prevents a stopping implementation from drifting
    away from deterministic prefix reconstruction.
    """

    normalized = normalized_ensemble_score(ensemble_score, alpha_l1_norm)
    y = np.asarray(y_true, dtype=np.float64)
    if normalized.shape != y.shape:
        raise ValueError("ensemble_score and y_true must have identical shapes.")
    return empirical_ramp_loss(y * normalized, mu)


def classification_error(y_true: Array, score: Array, threshold: float = 0.0) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    pred = np.where(s >= threshold, 1, -1).astype(np.int8)
    return float(np.mean(pred != y))
