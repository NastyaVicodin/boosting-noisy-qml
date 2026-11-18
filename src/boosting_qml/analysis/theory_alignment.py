"""Theory-aligned margin bounds, screening, and resource accounting.

This module is deliberately independent of the frozen experiment drivers.  It
contains deterministic postprocessing only and does not mutate historical
artifacts or reinterpret validation-accuracy stopping as margin stopping.
"""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from boosting_qml.losses.margins import empirical_ramp_loss, normalized_ensemble_score


PARAMETER_SHIFT_LEDGER_IDENTITY = "parameter_shift_equivalent_boosting.v1"
REFERENCE_PARAMETER_SHIFT_LEDGER_IDENTITY = "parameter_shift_equivalent_reference.v1"
SCREENING_SEED_NAMESPACES = {
    "real_adaboost": 710_000,
    "adaboost_r": 720_000,
}


def load_frozen_margin_target(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the human-frozen margin target artifact."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("margin target config is not frozen")
    mu = payload.get("mu")
    if mu is None or not (0.0 < float(mu) < 1.0):
        raise ValueError("frozen margin target must contain mu in (0, 1)")
    values = payload.get("epsilon_mar_by_depth_task")
    if not isinstance(values, Mapping) or len(values) != 25:
        raise ValueError("frozen margin target must contain exactly 25 epsilon values")
    for key, value in values.items():
        if not str(key).startswith("depth_") or not (0.0 < float(value) < 1.0):
            raise ValueError(f"invalid epsilon_mar entry {key!r}")
    source = Path(str(payload.get("calibration_source", "")))
    if not source.is_absolute():
        source = config_path.resolve().parents[1] / source
    if not source.is_file():
        raise ValueError("calibration source does not exist")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != payload.get("calibration_source_sha256"):
        raise ValueError("calibration source SHA-256 mismatch")
    if not payload.get("freeze_timestamp"):
        raise ValueError("frozen margin target lacks freeze_timestamp")
    attestation = str(payload.get("selection_attestation", ""))
    if "calibration-only" not in attestation.lower():
        raise ValueError("selection attestation must confirm calibration-only selection")
    forbidden = ("heldout", "test_accuracy", "boosting_runs")
    selection_inputs = json.dumps(payload.get("selection_inputs", []), sort_keys=True).lower()
    if any(token in selection_inputs for token in forbidden):
        raise ValueError("margin target selection inputs contain held-out/test data")
    return dict(payload)


def real_margin_contraction(mu: float, edge: float) -> float:
    """Return ``B_real(mu, edge)`` from the positive-margin theorem."""

    mu = float(mu)
    edge = float(edge)
    if not (0.0 < mu < 1.0):
        raise ValueError("mu must lie in (0, 1).")
    if not (0.0 <= edge < 1.0):
        raise ValueError("edge must lie in [0, 1).")
    log_b = 0.5 * (
        (1.0 - mu) * math.log1p(-edge)
        + (1.0 + mu) * math.log1p(edge)
    )
    return float(math.exp(log_b))


def real_contraction_holds(mu: float, edge: float) -> bool:
    """Whether the Real AdaBoost margin factor is strictly contractive."""

    return real_margin_contraction(mu, edge) < 1.0


def sufficient_rounds_from_contraction(epsilon_mar: float, contraction: float) -> int | None:
    """Return the theorem-sufficient round count, or ``None`` if vacuous."""

    epsilon_mar = float(epsilon_mar)
    contraction = float(contraction)
    if not (0.0 < epsilon_mar < 1.0):
        raise ValueError("epsilon_mar must lie in (0, 1).")
    if not (0.0 < contraction < 1.0):
        return None
    return int(math.ceil(math.log(1.0 / epsilon_mar) / -math.log(contraction)))


def real_uniform_edge_rounds(mu: float, epsilon_mar: float, eta: float) -> int | None:
    """Uniform-edge Real AdaBoost sufficient-round prediction."""

    return sufficient_rounds_from_contraction(
        epsilon_mar, real_margin_contraction(mu, eta)
    )


def real_trajectory_bound(mu: float, edges: Sequence[float]) -> np.ndarray:
    """Cumulative product of trajectory-dependent Real AdaBoost factors."""

    factors = np.asarray([real_margin_contraction(mu, r) for r in edges])
    return np.cumprod(factors, dtype=np.float64)


def adaboostr_q(mu: float, edge: float, h_star: float) -> float:
    """Return one-round AdaBoostR positive-margin progress ``q_t(mu)``."""

    mu = float(mu)
    edge = float(edge)
    h_star = float(h_star)
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("mu must be finite and positive.")
    if not (0.0 <= edge < 1.0):
        raise ValueError("edge must lie in [0, 1).")
    if h_star <= 0.0 or not np.isfinite(h_star):
        raise ValueError("h_star must be finite and positive.")
    return float(edge * edge - 2.0 * mu * np.arctanh(edge) / h_star)


def adaboostr_prefix_quantities(
    edges: Sequence[float], alphas: Sequence[float], mu: float
) -> dict[str, np.ndarray]:
    """Compute ``E_T``, ``A_T``, and ``E_T - 2 mu A_T`` for every prefix."""

    r = np.asarray(edges, dtype=np.float64)
    a = np.asarray(alphas, dtype=np.float64)
    if r.shape != a.shape:
        raise ValueError("edges and alphas must have identical shapes.")
    if np.any((r < 0.0) | (r >= 1.0)):
        raise ValueError("all edges must lie in [0, 1).")
    if np.any(a < 0.0):
        raise ValueError("all alphas must be nonnegative.")
    e_t = np.cumsum(r * r)
    a_t = np.cumsum(a)
    return {"E_T": e_t, "A_T": a_t, "Q_T": e_t - 2.0 * float(mu) * a_t}


def first_true_prefix(condition: Iterable[bool]) -> int | None:
    """Return the first one-based prefix satisfying a condition."""

    for index, value in enumerate(condition, start=1):
        if bool(value):
            return index
    return None


def first_ramp_target_prefix(
    labels: Sequence[float],
    ensemble_scores: Sequence[Sequence[float]],
    alpha_l1_norms: Sequence[float],
    *,
    mu: float,
    epsilon_mar: float,
) -> tuple[int | None, list[float]]:
    """Find the first accepted prefix attaining the empirical ramp target."""

    if len(ensemble_scores) != len(alpha_l1_norms):
        raise ValueError("score and normalization histories must have equal length.")
    y = np.asarray(labels, dtype=np.float64)
    losses: list[float] = []
    for score, norm in zip(ensemble_scores, alpha_l1_norms):
        normalized = normalized_ensemble_score(score, float(norm))
        if normalized.shape != y.shape:
            raise ValueError("each ensemble score must match labels.")
        losses.append(empirical_ramp_loss(y * normalized, mu))
    return first_true_prefix(value <= float(epsilon_mar) for value in losses), losses


def simultaneous_score_error_radius(
    *, n_shots: int, family_size: int, confidence_delta: float
) -> float:
    """Two-sided Hoeffding radius with an explicit union bound.

    Each physical shot is a value in ``{-1,+1}``.  ``family_size`` is the
    total number of score estimates covered simultaneously (examples times
    rounds, and any other explicitly included streams).
    """

    n_shots = int(n_shots)
    family_size = int(family_size)
    delta = float(confidence_delta)
    if n_shots <= 0 or family_size <= 0:
        raise ValueError("n_shots and family_size must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("confidence_delta must lie in (0, 1).")
    return float(min(2.0, math.sqrt(2.0 * math.log(2.0 * family_size / delta) / n_shots)))


def weak_score_intervals(
    estimated_scores: Sequence[float], error_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip simultaneous deterministic-mean score intervals to ``[-1, 1]``."""

    scores = np.asarray(estimated_scores, dtype=np.float64)
    radius = float(error_radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("error_radius must be finite and nonnegative.")
    return np.maximum(-1.0, scores - radius), np.minimum(1.0, scores + radius)


def edge_and_scale_interval(
    estimated_scores: Sequence[float],
    labels: Sequence[float],
    weights: Sequence[float],
    *,
    error_radius: float,
) -> dict[str, float]:
    """Conservative interval for ``|s|``, ``h_star``, and ``r=|s|/h_star``."""

    h = np.asarray(estimated_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    d = np.asarray(weights, dtype=np.float64)
    if h.shape != y.shape or h.shape != d.shape:
        raise ValueError("scores, labels, and weights must have identical shapes.")
    if np.any(d < 0.0) or not np.isclose(np.sum(d), 1.0, atol=1e-10):
        raise ValueError("weights must be nonnegative and sum to one.")
    lo, hi = weak_score_intervals(h, error_radius)
    signed_lo = np.where(y >= 0.0, lo, -hi)
    signed_hi = np.where(y >= 0.0, hi, -lo)
    s_lo = float(np.dot(d, signed_lo))
    s_hi = float(np.dot(d, signed_hi))
    abs_s_lo = 0.0 if s_lo <= 0.0 <= s_hi else min(abs(s_lo), abs(s_hi))
    abs_s_hi = max(abs(s_lo), abs(s_hi))
    h_lower = float(np.max(np.maximum(0.0, np.maximum(lo, -hi))))
    h_upper = float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
    if h_upper <= 0.0:
        r_lo = 0.0
        r_hi = 1.0
    else:
        r_lo = min(1.0, abs_s_lo / h_upper)
        r_hi = 1.0 if h_lower <= 0.0 else min(1.0, abs_s_hi / h_lower)
    return {
        "s_lower": s_lo,
        "s_upper": s_hi,
        "abs_s_lower": abs_s_lo,
        "abs_s_upper": abs_s_hi,
        "h_star_lower": h_lower,
        "h_star_upper": h_upper,
        "r_lower": r_lo,
        "r_upper": r_hi,
    }


def conservative_adaboostr_q_lower(
    *, mu: float, r_lower: float, r_upper: float, h_star_lower: float
) -> float | None:
    """Minimize ``q(r,h)`` over a conservative interval box.

    ``q`` is increasing in ``h`` for nonnegative ``r``, so the box minimum is
    attained at ``h_star_lower``.  We do not assume monotonicity in ``r``:
    endpoints and every real root of the derivative cubic inside the interval
    are evaluated.
    """

    mu = float(mu)
    lo = max(0.0, float(r_lower))
    hi = min(1.0 - 1e-12, float(r_upper))
    h_lo = float(h_star_lower)
    if h_lo <= 0.0 or lo > hi:
        return None
    candidates = [lo, hi]
    # q'(r)=0 iff r(1-r^2)=mu/h, i.e. r^3-r+mu/h=0.
    for root in np.roots([1.0, 0.0, -1.0, mu / h_lo]):
        if abs(float(np.imag(root))) <= 1e-10:
            value = float(np.real(root))
            if lo <= value <= hi:
                candidates.append(value)
    return float(min(adaboostr_q(mu, r, h_lo) for r in candidates))


def parameter_count(depth: int, n_qubits: int = 4) -> int:
    """Number of trainable parameters in the frozen reuploading ansatz."""

    depth = int(depth)
    n_qubits = int(n_qubits)
    if depth <= 0 or n_qubits <= 0:
        raise ValueError("depth and n_qubits must be positive.")
    return n_qubits * (4 * depth + 3)


def parameter_shift_equivalent_ledger(
    historical: Mapping[str, Any], *, depth: int, n_qubits: int = 4
) -> dict[str, Any]:
    """Reprice one disjoint straight-through round as physical parameter shift.

    The observed unshifted training-forward component is retained.  Only the
    additional ``2k`` shifted evaluations per training example are added; all
    monitors, edge estimates, confirmations, and diagnostics remain unchanged.
    """

    shots_per_expectation = float(historical.get("shots_per_expectation", 0.0) or 0.0)
    if shots_per_expectation <= 0.0:
        raw_total = float(historical.get("total_raw_shots", 0.0) or 0.0)
        eval_total = float(historical.get("total_model_evaluations", 0.0) or 0.0)
        if eval_total <= 0.0:
            raise ValueError("cannot infer shots_per_expectation from an empty ledger.")
        shots_per_expectation = raw_total / eval_total
    train_forward = float(historical["component_train_model_evaluations"])
    k = parameter_count(depth, n_qubits)
    components = {
        "component_train_forward": train_forward,
        "component_gradient_shifted": float(2 * k * train_forward),
        "component_training_monitor": float(historical["component_training_monitor_model_evaluations"]),
        "component_edge": float(historical["component_edge_model_evaluations"]),
        "component_validation_monitor": float(historical["component_validation_monitor_model_evaluations"]),
        "component_validation_confirm": float(historical["component_validation_confirm_model_evaluations"]),
        "component_diagnostic_test": float(historical.get("component_diagnostic_test_model_evaluations", 0.0) or 0.0),
    }
    total = float(sum(components.values()))
    result: dict[str, Any] = {
        "ledger_identity": PARAMETER_SHIFT_LEDGER_IDENTITY,
        "depth": int(depth),
        "n_qubits": int(n_qubits),
        "parameter_count": k,
        "parameter_shift_multiplier_1_plus_2k": 1 + 2 * k,
        "shots_per_expectation": shots_per_expectation,
        **components,
        "total_primary": total,
        "total_primary_physical_shots": total * shots_per_expectation,
    }
    validate_parameter_shift_equivalent_ledger(result)
    return result


def parameter_shift_reference_ledger(
    *,
    unshifted_training_evaluations: int,
    depth: int,
    target_monitor_evaluations: int,
    shots_per_expectation: int = 1000,
    n_qubits: int = 4,
) -> dict[str, Any]:
    """Project an ideal reference prefix onto the established PS ledger."""

    unshifted = int(unshifted_training_evaluations)
    monitor = int(target_monitor_evaluations)
    shots = int(shots_per_expectation)
    if unshifted < 0 or monitor < 0 or shots <= 0:
        raise ValueError("reference ledger counts must be nonnegative and shots positive")
    k = parameter_count(depth, n_qubits)
    shifted = 2 * k * unshifted
    total = unshifted + shifted + monitor
    return {
        "ledger_identity": REFERENCE_PARAMETER_SHIFT_LEDGER_IDENTITY,
        "depth": int(depth),
        "n_qubits": int(n_qubits),
        "parameter_count": k,
        "parameter_shift_multiplier_1_plus_2k": 1 + 2 * k,
        "component_train_forward": unshifted,
        "component_gradient_shifted": shifted,
        "component_target_monitor": monitor,
        "total_primary": total,
        "shots_per_expectation": shots,
        "total_primary_physical_shots": total * shots,
    }


def validate_parameter_shift_equivalent_ledger(ledger: Mapping[str, Any]) -> None:
    """Require exact additivity of the projected disjoint component ledger."""

    fields = (
        "component_train_forward",
        "component_gradient_shifted",
        "component_training_monitor",
        "component_edge",
        "component_validation_monitor",
        "component_validation_confirm",
        "component_diagnostic_test",
    )
    values = [float(ledger[field]) for field in fields]
    if any(value < 0.0 for value in values):
        raise ValueError("parameter-shift-equivalent ledger contains a negative cost.")
    total = float(ledger["total_primary"])
    if not math.isclose(sum(values), total, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("parameter-shift-equivalent component ledger is non-additive.")
    shots = total * float(ledger["shots_per_expectation"])
    if not math.isclose(
        shots,
        float(ledger["total_primary_physical_shots"]),
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError("parameter-shift-equivalent shot total is non-additive.")


def screening_seed(method: str, *, depth: int, task_index: int, noise_index: int) -> int:
    """Return a deterministic screening-only seed disjoint from evaluation seeds."""

    if method not in SCREENING_SEED_NAMESPACES:
        raise ValueError(f"unknown screening method {method!r}.")
    for name, value in {
        "depth": depth,
        "task_index": task_index,
        "noise_index": noise_index,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    return int(
        SCREENING_SEED_NAMESPACES[method]
        + int(depth) * 1_000
        + int(task_index) * 10
        + int(noise_index)
    )


def screening_decision(
    boost_cost: float | None,
    pec_cost: float | None,
    *,
    overlap_or_vacuous: bool = False,
) -> str:
    """Return the three-way screening decision without forcing a binary call."""

    if overlap_or_vacuous or boost_cost is None or pec_cost is None:
        return "inconclusive"
    if not (np.isfinite(boost_cost) and np.isfinite(pec_cost)):
        return "inconclusive"
    return "boosting_favored" if float(boost_cost) < float(pec_cost) else "PEC_favored"


def hierarchical_task_cell_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    task_field: str = "task_index",
    success_field: str = "target_reached",
) -> dict[str, Any]:
    """Median within task, then median across task-level medians.

    Censored rows contribute to the raw success denominator but never enter a
    successful cost median.  This prevents stochastic runs from being treated
    as independent teacher-task replicates.
    """

    grouped: dict[int, list[float]] = {}
    attempts = len(rows)
    successes = 0
    for row in rows:
        if bool(row.get(success_field, False)):
            value = row.get(value_field)
            if value is None or not np.isfinite(float(value)):
                raise ValueError("successful row lacks a finite value")
            grouped.setdefault(int(row[task_field]), []).append(float(value))
            successes += 1
    task_medians = {
        task: float(np.median(values)) for task, values in sorted(grouped.items())
    }
    return {
        "attempts": attempts,
        "successes": successes,
        "right_censored": attempts - successes,
        "task_success_counts": {task: len(values) for task, values in sorted(grouped.items())},
        "task_medians": task_medians,
        "cell_median_of_task_medians": (
            None if not task_medians else float(np.median(list(task_medians.values())))
        ),
        "aggregation": "median_within_teacher_task_then_median_across_teacher_tasks",
    }
