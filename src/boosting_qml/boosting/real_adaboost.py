"""Real AdaBoost with confidence-rated weak predictions.

This implementation keeps the outer boosting loop in Python while using JAX
for the per-round array bookkeeping when JAX is available. When JAX is not
installed, it falls back to the original NumPy behavior so the lightweight
classical test environment still works.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, List, Mapping, Optional

import numpy as np

from boosting_qml.analysis.round_scaling import (
    estimate_round_budget_real_adaboost,
    estimate_running_convergence_diagnostics,
    format_running_convergence_diagnostics,
)
from boosting_qml.config import BoostingStopTargetConfig
from boosting_qml.boosting.stopping import (
    checkpoint_stop_target_matches,
    fixed_train_margin_loss_decision,
)
from boosting_qml.losses.margins import (
    empirical_margin_loss,
    normalized_empirical_ramp_loss,
    normalized_margin,
)
from boosting_qml.pec import (
    empty_shot_accounting_dict,
    shot_accounting_to_dict,
    sum_shot_accounting_dicts,
    validate_boosting_component_accounting,
)
from boosting_qml.rng import boosting_round_rng_seeds, stable_int_seed
from boosting_qml.types import Array, BinaryDataset, BoostingRoundResult, BoostingStopSummary
from boosting_qml.weak_learners.base import BinaryWeakLearner
from boosting_qml.weak_learners.classical_linear import WeightedLinearWeakLearner

try:  # pragma: no cover - exercised in JAX-capable environments
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf as _jax_erf, logsumexp as _jax_logsumexp

    _JAX_AVAILABLE = True
    _JAX_FLOAT = jnp.float32
except Exception:  # pragma: no cover - exercised by local lightweight tests
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _jax_erf = None  # type: ignore[assignment]
    _jax_logsumexp = None  # type: ignore[assignment]
    _JAX_AVAILABLE = False
    _JAX_FLOAT = None

_EPS = np.finfo(np.float64).eps
_MIN_ALPHA_L1 = 1e-12


def _logsumexp(x: Array) -> float:
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def _np_clip_scores(scores: Array) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=np.float64), -1.0, 1.0)


def _np_edge_and_boost(
    example_weights: Array,
    y_true: Array,
    train_scores_raw: Array,
    clip_edge: float,
) -> tuple[float, float, np.ndarray, float]:
    weights = np.asarray(example_weights, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    scores_raw = np.asarray(train_scores_raw, dtype=np.float64)

    edge_raw = float(np.dot(weights, y * scores_raw))
    weak_sign = -1.0 if edge_raw < 0.0 else 1.0
    edge_t = min(abs(edge_raw), 1.0 - clip_edge)
    train_scores = weak_sign * scores_raw
    boost_coef_t = float(0.5 * np.log((1.0 + edge_t) / (1.0 - edge_t + _EPS)))
    return edge_t, weak_sign, train_scores, boost_coef_t


def _np_update_weights(
    example_weights: Array,
    y_true: Array,
    train_scores: Array,
    boost_coef_t: float,
) -> np.ndarray:
    weights = np.asarray(example_weights, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(train_scores, dtype=np.float64)

    log_weights = np.log(weights + _EPS) - float(boost_coef_t) * y * scores
    log_weights -= _logsumexp(log_weights)
    return np.exp(log_weights)


def _np_update_ensemble(
    ensemble_score: Array,
    weak_scores: Array,
    boost_coef_t: float,
) -> np.ndarray:
    return np.asarray(ensemble_score, dtype=np.float64) + float(boost_coef_t) * np.asarray(
        weak_scores,
        dtype=np.float64,
    )


def _np_dataset_metrics(
    y_true: Array,
    ensemble_score: Array,
    margin_threshold: float,
    alpha_l1_norm: float,
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(ensemble_score, dtype=np.float64)
    pred = np.where(score >= 0.0, 1.0, -1.0)
    error = float(np.mean(pred != y))
    accuracy = float(1.0 - error)
    margin_values = normalized_margin(y, score, alpha_l1_norm=max(float(alpha_l1_norm), _MIN_ALPHA_L1))
    margin_loss = float(empirical_margin_loss(margin_values, margin_threshold))
    return error, accuracy, margin_loss



def _np_smooth_hinge_loss_from_scores(
    example_weights: Array,
    y_true: Array,
    signed_scores: Array,
    sigma: float,
) -> float:
    weights = np.asarray(example_weights, dtype=np.float64)
    weights = np.clip(weights, _EPS, None)
    weights = weights / np.sum(weights)
    margins = np.asarray(y_true, dtype=np.float64) * np.asarray(signed_scores, dtype=np.float64)
    sigma = max(float(sigma), _EPS)
    v = (1.0 - margins) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(v / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-(v * v) / 2.0)
    losses = cdf * (1.0 - margins) + pdf * sigma
    return float(np.dot(weights, losses))


def _example_weights_for_fit(learner: BinaryWeakLearner, example_weights: Array, *, use_jax: bool) -> Array:
    if use_jax and bool(getattr(learner, "accepts_jax_example_weights", False)):
        return example_weights
    return np.asarray(example_weights, dtype=np.float64)


def _training_phase(learner: BinaryWeakLearner, audit_role: str = "edge"):
    """Context that attributes per-round diagnostic predictions to the train budget.

    Returns the learner's ``prediction_phase("train")`` context when supported,
    otherwise a no-op so learners without phase-aware accounting are unaffected.
    """
    if hasattr(learner, "audit_prediction_phase"):
        return learner.audit_prediction_phase(audit_role)
    phase_fn = getattr(learner, "prediction_phase", None)
    if callable(phase_fn):
        return phase_fn("train")
    return nullcontext()


def _weak_scores_backend(
    learner: BinaryWeakLearner, X: Array, rng, *, use_jax: bool, prefer_jax: bool = True
):
    """Weak scores as a device ``jax.Array`` when the learner supports in-graph
    prediction (Opt 1) and ``prefer_jax`` is set, else a float64 NumPy array.

    Keeping device arrays on device lets the per-round JAX kernel consume them
    without a device->host->device round-trip. Setting ``prefer_jax=False``
    forces the host-materialized ``predict_score`` path (used to benchmark the
    pre-Opt-1 baseline from a single checkout). Learners lacking
    ``predict_scores_jax`` always fall back to ``predict_score``.
    """
    if use_jax and prefer_jax and bool(getattr(learner, "supports_jax_prediction", False)):
        predict_jax = getattr(learner, "predict_scores_jax", None)
        if callable(predict_jax):
            return predict_jax(X, rng=rng)
    return np.asarray(learner.predict_score(X, rng=rng), dtype=np.float64)


def _deterministic_mean_scores(learner: BinaryWeakLearner, X: Array) -> np.ndarray:
    predict = getattr(learner, "predict_deterministic_mean_score", None)
    if not callable(predict):
        raise TypeError(
            "ensemble patience requires a weak learner with "
            "predict_deterministic_mean_score"
        )
    return _np_clip_scores(predict(X))

if _JAX_AVAILABLE:  # pragma: no branch - defined once when JAX imports cleanly

    @jax.jit
    def _jax_clip_scores(scores: Any) -> Any:
        return jnp.clip(scores, -1.0, 1.0)


    @jax.jit
    def _jax_edge_and_boost(
        example_weights: Any,
        y_true: Any,
        train_scores_raw: Any,
        clip_edge: float,
    ) -> tuple[Any, Any, Any, Any]:
        eps = jnp.asarray(_EPS, dtype=train_scores_raw.dtype)
        clip_edge_arr = jnp.asarray(clip_edge, dtype=train_scores_raw.dtype)
        edge_raw = jnp.dot(example_weights, y_true * train_scores_raw)
        weak_sign = jnp.where(edge_raw < 0.0, -1.0, 1.0).astype(train_scores_raw.dtype)
        edge_t = jnp.minimum(jnp.abs(edge_raw), 1.0 - clip_edge_arr)
        train_scores = weak_sign * train_scores_raw
        boost_coef_t = 0.5 * jnp.log((1.0 + edge_t) / (1.0 - edge_t + eps))
        return edge_t, weak_sign, train_scores, boost_coef_t


    @jax.jit
    def _jax_update_weights(
        example_weights: Any,
        y_true: Any,
        train_scores: Any,
        boost_coef_t: Any,
    ) -> Any:
        eps = jnp.asarray(_EPS, dtype=train_scores.dtype)
        log_weights = jnp.log(example_weights + eps) - boost_coef_t * y_true * train_scores
        log_weights = log_weights - _jax_logsumexp(log_weights)
        return jnp.exp(log_weights)


    @jax.jit
    def _jax_update_ensemble(
        ensemble_score: Any,
        weak_scores: Any,
        boost_coef_t: Any,
    ) -> Any:
        return ensemble_score + boost_coef_t * weak_scores


    @jax.jit
    def _jax_dataset_metrics(
        y_true: Any,
        ensemble_score: Any,
        margin_threshold: float,
        alpha_l1_norm: Any,
    ) -> tuple[Any, Any, Any]:
        eps = jnp.asarray(_EPS, dtype=ensemble_score.dtype)
        min_alpha = jnp.asarray(_MIN_ALPHA_L1, dtype=ensemble_score.dtype)
        threshold = jnp.asarray(margin_threshold, dtype=ensemble_score.dtype)
        pred = jnp.where(ensemble_score >= 0.0, 1.0, -1.0)
        error = jnp.mean(pred != y_true)
        accuracy = 1.0 - error
        denom = jnp.maximum(alpha_l1_norm, min_alpha) + eps
        margins = y_true * ensemble_score / denom
        losses = jnp.where(
            margins >= threshold,
            0.0,
            jnp.where(margins >= 0.0, 1.0 - margins / threshold, 1.0),
        )
        margin_loss = jnp.mean(losses)
        return error, accuracy, margin_loss




    @jax.jit
    def _jax_train_round(
        example_weights: Any,
        y_true: Any,
        train_scores_raw: Any,
        ensemble_score: Any,
        alpha_l1_norm: Any,
        margin_threshold: float,
        clip_edge: float,
        surrogate_sigma: float,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        eps = jnp.asarray(_EPS, dtype=train_scores_raw.dtype)
        min_alpha = jnp.asarray(_MIN_ALPHA_L1, dtype=train_scores_raw.dtype)
        threshold = jnp.asarray(margin_threshold, dtype=train_scores_raw.dtype)
        clip_edge_arr = jnp.asarray(clip_edge, dtype=train_scores_raw.dtype)

        train_scores_raw = jnp.clip(train_scores_raw, -1.0, 1.0)
        edge_raw = jnp.dot(example_weights, y_true * train_scores_raw)
        weak_sign = jnp.where(edge_raw < 0.0, -1.0, 1.0).astype(train_scores_raw.dtype)
        edge_t = jnp.minimum(jnp.abs(edge_raw), 1.0 - clip_edge_arr)
        train_scores = weak_sign * train_scores_raw
        boost_coef_t = 0.5 * jnp.log((1.0 + edge_t) / (1.0 - edge_t + eps))

        log_weights = jnp.log(example_weights + eps) - boost_coef_t * y_true * train_scores
        log_weights = log_weights - _jax_logsumexp(log_weights)
        updated_weights = jnp.exp(log_weights)

        updated_ensemble = ensemble_score + boost_coef_t * train_scores
        updated_alpha_l1 = alpha_l1_norm + jnp.abs(boost_coef_t)
        pred = jnp.where(updated_ensemble >= 0.0, 1.0, -1.0)
        error = jnp.mean(pred != y_true)
        denom = jnp.maximum(updated_alpha_l1, min_alpha) + eps
        margins = y_true * updated_ensemble / denom
        losses = jnp.where(
            margins >= threshold,
            0.0,
            jnp.where(margins >= 0.0, 1.0 - margins / threshold, 1.0),
        )
        margin_loss = jnp.mean(losses)

        sigma = jnp.maximum(jnp.asarray(surrogate_sigma, dtype=train_scores_raw.dtype), eps)
        weak_margins = y_true * train_scores
        v = (1.0 - weak_margins) / sigma
        smooth_cdf = 0.5 * (1.0 + _jax_erf(v / jnp.sqrt(2.0)))
        smooth_pdf = (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-(v * v) / 2.0)
        smooth_losses = smooth_cdf * (1.0 - weak_margins) + smooth_pdf * sigma
        w_norm = example_weights / (jnp.sum(example_weights) + eps)
        smooth_hinge_loss = jnp.dot(w_norm, smooth_losses)
        surrogate_edge_lower_bound = 1.0 - smooth_hinge_loss
        return (
            edge_t,
            weak_sign,
            train_scores,
            boost_coef_t,
            updated_weights,
            updated_ensemble,
            updated_alpha_l1,
            error,
            margin_loss,
            smooth_hinge_loss,
            surrogate_edge_lower_bound,
        )


def _serialize_weak_learner_state(learner: BinaryWeakLearner) -> dict[str, Any]:
    state = dict(learner.state_dict())
    learner_type = state.get('weak_learner_type')
    if learner_type is None:
        class_name = type(learner).__name__
        if class_name == 'WeightedLinearWeakLearner':
            learner_type = 'weighted_linear'
        elif class_name == 'WeightedPQCBinaryWeakLearner':
            learner_type = 'weighted_pqc_binary'
        else:
            learner_type = class_name
    state['weak_learner_type'] = str(learner_type)
    return state


def _deserialize_weak_learner_state(state: Mapping[str, Any]) -> BinaryWeakLearner:
    learner_type = str(state.get('weak_learner_type', '')).strip().lower()

    is_linear_state = 'coef' in state and 'intercept' in state and 'ridge_lambda' in state
    is_pqc_state = 'params' in state and 'n_qubits' in state and 'n_layers' in state

    if learner_type in {'', 'weighted_linear', 'weightedlinearweaklearner', 'linear'}:
        if is_linear_state:
            return WeightedLinearWeakLearner.from_state_dict(state)

    if learner_type in {'weighted_pqc_binary', 'weightedpqcbinaryweaklearner', 'pqc', 'pqc_binary'}:
        try:
            from boosting_qml.weak_learners.pqc_binary import WeightedPQCBinaryWeakLearner
        except Exception as exc:
            raise ImportError(
                'Loading a PQC weak learner requires optional quantum dependencies '
                '(jax/tensorcircuit and error_mitigation_PEC imports).'
            ) from exc
        return WeightedPQCBinaryWeakLearner.from_state_dict(state)

    if learner_type == '' and is_pqc_state:
        try:
            from boosting_qml.weak_learners.pqc_binary import WeightedPQCBinaryWeakLearner
        except Exception as exc:
            raise ImportError(
                'Detected PQC weak-learner checkpoint fields but could not import '
                'WeightedPQCBinaryWeakLearner dependencies.'
            ) from exc
        return WeightedPQCBinaryWeakLearner.from_state_dict(state)

    if learner_type == '':
        raise ValueError(
            'Could not infer weak learner type from checkpoint state. '
            "Expected linear fields ('coef','intercept','ridge_lambda') or "
            "PQC fields ('params','n_qubits','n_layers')."
        )

    raise ValueError(f"Unsupported weak_learner_type='{learner_type}' in checkpoint.")


def _learner_shot_accounting(learner: BinaryWeakLearner) -> dict[str, Any]:
    accounting_fn = getattr(learner, "shot_accounting_summary", None)
    if callable(accounting_fn):
        return shot_accounting_to_dict(accounting_fn())
    return empty_shot_accounting_dict(source=type(learner).__name__)


def _shot_budget_consumed(accounting: Mapping[str, Any]) -> int:
    return int(math.ceil(float(accounting.get("total_raw_shots", 0.0) or 0.0)))


_SHOT_ACCOUNTING_DELTA_KEYS = (
    "train_model_evaluations",
    "inference_model_evaluations",
    "total_model_evaluations",
    "train_effective_shots",
    "inference_effective_shots",
    "total_effective_shots",
    "train_raw_shots",
    "inference_raw_shots",
    "total_raw_shots",
    "train_trajectory_evaluations",
    "inference_trajectory_evaluations",
    "total_trajectory_evaluations",
)


def _shot_accounting_delta(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    after_payload = shot_accounting_to_dict(after)
    before_payload = shot_accounting_to_dict(before)
    delta = empty_shot_accounting_dict(source=source)
    delta["n_shots_eff_per_expectation"] = max(
        int(after_payload.get("n_shots_eff_per_expectation", 0) or 0),
        int(before_payload.get("n_shots_eff_per_expectation", 0) or 0),
    )
    delta["raw_shot_overhead"] = max(
        float(after_payload.get("raw_shot_overhead", 1.0) or 1.0),
        float(before_payload.get("raw_shot_overhead", 1.0) or 1.0),
    )
    delta["trajectory_multiplier"] = max(
        int(after_payload.get("trajectory_multiplier", 1) or 1),
        int(before_payload.get("trajectory_multiplier", 1) or 1),
    )
    for key in _SHOT_ACCOUNTING_DELTA_KEYS:
        delta[key] = (after_payload.get(key, 0) or 0) - (before_payload.get(key, 0) or 0)
    return delta


def _annotate_boosting_component_accounting(
    shot_accounting: Mapping[str, Any],
    *,
    fit_accounting: Mapping[str, Any],
    edge_accounting: Mapping[str, Any],
    validation_monitor_accounting: Mapping[str, Any],
    validation_confirm_accounting: Mapping[str, Any],
    train_margin_confirm_accounting: Optional[Mapping[str, Any]] = None,
    diagnostic_test_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    payload = shot_accounting_to_dict(shot_accounting)
    fit_payload = shot_accounting_to_dict(fit_accounting)
    edge_payload = shot_accounting_to_dict(edge_accounting)
    validation_monitor_payload = shot_accounting_to_dict(validation_monitor_accounting)
    validation_confirm_payload = shot_accounting_to_dict(validation_confirm_accounting)
    train_margin_confirm_payload = shot_accounting_to_dict(
        empty_shot_accounting_dict(source="train_margin_confirm")
        if train_margin_confirm_accounting is None
        else train_margin_confirm_accounting
    )
    diagnostic_test_payload = shot_accounting_to_dict(diagnostic_test_accounting)
    shots_per_expectation = float(payload.get("shots_per_expectation", 0.0) or 0.0)
    raw_overhead = float(payload.get("raw_shot_overhead", 1.0) or 1.0)
    unshifted_loss_evaluations = int(payload.get("unshifted_loss_evaluations", 0) or 0)
    shifted_gradient_evaluations = int(payload.get("shifted_gradient_evaluations", 0) or 0)
    training_monitor_evaluations = int(payload.get("training_monitor_evaluations", 0) or 0)
    training_monitor_raw_shots = float(
        training_monitor_evaluations * shots_per_expectation * raw_overhead
    )
    edge_model_evaluations = int(
        edge_payload.get("total_model_evaluations", 0) or 0
    )
    validation_monitor_model_evaluations = int(
        validation_monitor_payload.get("total_model_evaluations", 0) or 0
    )
    validation_confirm_model_evaluations = int(
        validation_confirm_payload.get("total_model_evaluations", 0) or 0
    )
    train_margin_confirm_model_evaluations = int(
        train_margin_confirm_payload.get("total_model_evaluations", 0) or 0
    )
    diagnostic_test_model_evaluations = int(
        diagnostic_test_payload.get("total_model_evaluations", 0) or 0
    )
    edge_raw_shots = float(edge_payload.get("total_raw_shots", 0.0) or 0.0)
    validation_monitor_raw_shots = float(
        validation_monitor_payload.get("total_raw_shots", 0.0) or 0.0
    )
    validation_confirm_raw_shots = float(
        validation_confirm_payload.get("total_raw_shots", 0.0) or 0.0
    )
    train_margin_confirm_raw_shots = float(
        train_margin_confirm_payload.get("total_raw_shots", 0.0) or 0.0
    )
    diagnostic_test_raw_shots = float(
        diagnostic_test_payload.get("total_raw_shots", 0.0) or 0.0
    )
    train_raw_shots = float(payload.get("gradient_unshifted_shots", 0.0) or 0.0)
    if train_raw_shots == 0.0 and unshifted_loss_evaluations == 0:
        train_raw_shots = max(
            0.0,
            float(payload.get("total_raw_shots", 0.0) or 0.0)
            - float(payload.get("gradient_shifted_shots", 0.0) or 0.0)
            - training_monitor_raw_shots
            - edge_raw_shots
            - validation_monitor_raw_shots
            - validation_confirm_raw_shots
            - train_margin_confirm_raw_shots
            - diagnostic_test_raw_shots,
        )
    train_model_evaluations = unshifted_loss_evaluations
    if train_model_evaluations == 0 and shifted_gradient_evaluations == 0:
        train_model_evaluations = max(
            0,
            int(payload.get("total_model_evaluations", 0) or 0)
            - training_monitor_evaluations
            - edge_model_evaluations
            - validation_monitor_model_evaluations
            - validation_confirm_model_evaluations
            - train_margin_confirm_model_evaluations
            - diagnostic_test_model_evaluations,
        )
    payload.update(
        {
            "boosting_component_ledger_basis": (
                "disjoint_actual_learner_accounting_deltas.v2"
            ),
            "component_train_model_evaluations": train_model_evaluations,
            "component_gradient_model_evaluations": shifted_gradient_evaluations,
            "component_training_monitor_model_evaluations": training_monitor_evaluations,
            "component_edge_model_evaluations": edge_model_evaluations,
            "component_validation_monitor_model_evaluations": (
                validation_monitor_model_evaluations
            ),
            "component_validation_confirm_model_evaluations": (
                validation_confirm_model_evaluations
            ),
            "component_train_margin_confirm_model_evaluations": (
                train_margin_confirm_model_evaluations
            ),
            "component_diagnostic_test_model_evaluations": (
                diagnostic_test_model_evaluations
            ),
            "component_train_raw_shots": train_raw_shots,
            "component_gradient_raw_shots": float(payload.get("gradient_shifted_shots", 0.0) or 0.0),
            "component_training_monitor_raw_shots": training_monitor_raw_shots,
            "component_edge_raw_shots": edge_raw_shots,
            "component_validation_monitor_raw_shots": (
                validation_monitor_raw_shots
            ),
            "component_validation_confirm_raw_shots": (
                validation_confirm_raw_shots
            ),
            "component_train_margin_confirm_raw_shots": train_margin_confirm_raw_shots,
            "component_diagnostic_test_raw_shots": diagnostic_test_raw_shots,
        }
    )
    validate_boosting_component_accounting(payload)
    return payload


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json_atomic(path_obj: Path, payload: Mapping[str, Any]) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_name(path_obj.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp_path.replace(path_obj)


_ROUND_ARRAY_FIELDS = {
    "train_scores",
    "train_ensemble_score",
    "example_weights_before_update",
    "example_weights_after_update",
    "validation_scores",
    "validation_ensemble_score",
    "test_scores",
    "train_margin_confirmation_ensemble_score",
}


def _serialize_round_result(result: BoostingRoundResult) -> dict[str, Any]:
    return asdict(result)


def _deserialize_round_result(state: Mapping[str, Any]) -> BoostingRoundResult:
    values = dict(state)
    for name in _ROUND_ARRAY_FIELDS:
        if name in values and values[name] is not None:
            values[name] = np.asarray(values[name], dtype=np.float64)
    return BoostingRoundResult(**values)


def _round_budget_diagnostic_suffix(
    *,
    round_results: List[BoostingRoundResult],
    y_train: np.ndarray,
    margin_threshold: float,
    target_margin_loss: float,
) -> str:
    if not round_results:
        return ""
    try:
        rows = estimate_running_convergence_diagnostics(
            round_results=round_results,
            y_train=y_train,
            margin_threshold=float(margin_threshold),
            target_margin_loss=float(target_margin_loss),
            algorithm="real_adaboost",
        )
    except Exception as exc:  # pragma: no cover - diagnostic logging must not stop training.
        return f" T*=unavailable({type(exc).__name__})"
    if not rows:
        return ""
    return " " + format_running_convergence_diagnostics(rows[-1])


def _planned_round_budget_real(
    *,
    edge_lower_bound: float,
    margin_threshold: float,
    target_margin_loss: float,
    cap_rounds: int,
    warmup_rounds: int,
) -> int:
    if cap_rounds <= 0:
        raise ValueError('cap_rounds must be positive.')
    if not (0.0 < float(target_margin_loss) < 1.0):
        return int(cap_rounds)
    if not np.isfinite(edge_lower_bound) or edge_lower_bound <= 0.0:
        return int(cap_rounds)

    eta = min(max(float(edge_lower_bound), 1e-12), 1.0 - 1e-12)
    try:
        planned_rounds = estimate_round_budget_real_adaboost(
            eta=eta,
            margin_threshold=float(margin_threshold),
            target_margin_loss=float(target_margin_loss),
        )
    except ValueError:
        return int(cap_rounds)

    if not np.isfinite(planned_rounds):
        return int(cap_rounds)
    return int(max(warmup_rounds, min(cap_rounds, int(math.ceil(planned_rounds)))))


@dataclass
class RealAdaBoost:
    """Real AdaBoost implementation isolated from AdaBoost.R logic."""

    n_rounds: int
    margin_threshold: float = 0.2
    clip_edge: float = 1e-12
    surrogate_sigma: float = 0.5
    use_in_graph_prediction: bool = True  # Opt 1: keep per-round scores on device
    verbose: bool = False
    log_every_rounds: int = 1
    checkpoint_path: Optional[str] = None
    checkpoint_every_rounds: int = 0
    checkpoint_metadata: Optional[dict[str, Any]] = None
    test_evaluation_policy: str = "final_only"
    validation_confirmation_policy: str = "every_round_independent_stream"
    train_margin_confirmation_policy: str = "none"
    ensemble_patience_rounds: Optional[int] = None
    ensemble_patience_min_delta: float = 0.0

    weak_learners: List[BinaryWeakLearner] = field(default_factory=list, init=False)
    weak_learner_signs: List[float] = field(default_factory=list, init=False)
    boost_coefs: List[float] = field(default_factory=list, init=False)
    round_results: List[BoostingRoundResult] = field(default_factory=list, init=False)
    stopping_summary: Optional[BoostingStopSummary] = field(default=None, init=False)
    resumable_state: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    _restored_checkpoint: bool = field(default=False, init=False, repr=False)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[RealAdaBoost] {message}", flush=True)

    def _should_log_round(self, round_idx: int, *, force: bool = False) -> bool:
        if not self.verbose:
            return False
        every = max(1, int(self.log_every_rounds))
        return bool(force or round_idx == 1 or round_idx >= int(self.n_rounds) or round_idx % every == 0)

    def _checkpoint_metadata_payload(self, *, reason: str) -> dict[str, Any]:
        metadata = dict(self.checkpoint_metadata or {})
        metadata.update(
            {
                'checkpoint_reason': str(reason),
                'completed_rounds': int(len(self.round_results)),
                'n_rounds_cap': int(self.n_rounds),
            }
        )
        return metadata

    def _maybe_write_checkpoint(self, round_idx: int, *, reason: str = 'round_completed', force: bool = False) -> None:
        if not self.checkpoint_path:
            return
        every = int(self.checkpoint_every_rounds)
        if every <= 0 and not force:
            return
        if not force and int(round_idx) % every != 0:
            return
        self.save(self.checkpoint_path, checkpoint_metadata=self._checkpoint_metadata_payload(reason=reason))

    def _latest_resume_state(self) -> Optional[dict[str, Any]]:
        return None if self.resumable_state is None else dict(self.resumable_state)

    def _validate_restored_state(
        self,
        *,
        train_data: BinaryDataset,
        validation_data: Optional[BinaryDataset],
        test_data: Optional[BinaryDataset],
        rng_seed: int,
        stop_target: Optional[BoostingStopTargetConfig],
    ) -> int:
        completed = len(self.round_results)
        lengths = {
            "weak_learners": len(self.weak_learners),
            "weak_learner_signs": len(self.weak_learner_signs),
            "boost_coefs": len(self.boost_coefs),
        }
        if any(value != completed for value in lengths.values()):
            raise ValueError(
                "Incompatible RealAdaBoost checkpoint: accepted-round state "
                f"lengths disagree (round_results={completed}, {lengths})."
            )
        if [row.round_index for row in self.round_results] != list(range(1, completed + 1)):
            raise ValueError(
                "Incompatible RealAdaBoost checkpoint: round indices are not a "
                "contiguous one-based prefix."
            )
        if completed == 0:
            return 1
        if not isinstance(self.resumable_state, Mapping):
            raise ValueError(
                "Incompatible RealAdaBoost checkpoint: completed rounds require "
                "latest_resume_state."
            )
        state = self.resumable_state
        if not checkpoint_stop_target_matches(
            state.get("stop_target"),
            None if stop_target is None else asdict(stop_target),
        ):
            raise ValueError(
                "Incompatible RealAdaBoost checkpoint: validation stopping target changed."
            )
        if int(state.get("round_index", -1)) != completed:
            raise ValueError("Incompatible RealAdaBoost checkpoint: resumable round differs.")
        if int(state.get("base_rng_seed", -1)) != int(rng_seed):
            raise ValueError("Incompatible RealAdaBoost checkpoint: boosting RNG seed changed.")
        expected_sizes = {
            "train_n_samples": train_data.n_samples,
            "validation_n_samples": None if validation_data is None else validation_data.n_samples,
            "test_n_samples": None if test_data is None else test_data.n_samples,
        }
        for name, expected in expected_sizes.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"Incompatible RealAdaBoost checkpoint: dataset size changed for {name}."
                )
        return completed + 1

    def fit(
        self,
        train_data: BinaryDataset,
        weak_learner_factory: Callable[[], BinaryWeakLearner],
        rng_seed: int = 0,
        validation_data: Optional[BinaryDataset] = None,
        test_data: Optional[BinaryDataset] = None,
        stop_target: Optional[BoostingStopTargetConfig] = None,
    ) -> 'RealAdaBoost':
        resume_checkpoint = bool(self._restored_checkpoint)
        if not resume_checkpoint:
            self.weak_learners.clear()
            self.weak_learner_signs.clear()
            self.boost_coefs.clear()
            self.round_results.clear()
            self.stopping_summary = None
            self.resumable_state = None

        if self.n_rounds <= 0:
            raise ValueError('n_rounds must be positive.')
        fixed_train_margin_stop = bool(
            stop_target is not None
            and stop_target.stop_mode == 'fixed_train_margin_loss'
        )
        if stop_target is not None and not fixed_train_margin_stop and validation_data is None:
            raise ValueError('validation_data is required when stop_target is provided.')
        valid_test_policies = {"final_only", "per_round_diagnostic_holdout"}
        test_evaluation_policy = str(self.test_evaluation_policy)
        if test_evaluation_policy not in valid_test_policies:
            raise ValueError(
                f"test_evaluation_policy must be one of {sorted(valid_test_policies)}, "
                f"got {test_evaluation_policy!r}."
            )
        valid_confirmation_policies = {
            "every_round_independent_stream",
            "monitor_then_confirm_on_success_independent_stream",
        }
        if self.validation_confirmation_policy not in valid_confirmation_policies:
            raise ValueError(
                "validation_confirmation_policy must be one of "
                f"{sorted(valid_confirmation_policies)}, got "
                f"{self.validation_confirmation_policy!r}."
            )
        valid_train_margin_confirmation_policies = {
            "none",
            "monitor_then_confirm_on_success_independent_stream",
        }
        if self.train_margin_confirmation_policy not in valid_train_margin_confirmation_policies:
            raise ValueError(
                "train_margin_confirmation_policy must be one of "
                f"{sorted(valid_train_margin_confirmation_policies)}, got "
                f"{self.train_margin_confirmation_policy!r}."
            )
        if self.train_margin_confirmation_policy != "none" and not fixed_train_margin_stop:
            raise ValueError(
                "training-margin confirmation requires fixed_train_margin_loss stopping."
            )
        if self.ensemble_patience_rounds is not None and int(self.ensemble_patience_rounds) <= 0:
            raise ValueError("ensemble_patience_rounds must be positive when enabled.")
        if not math.isfinite(float(self.ensemble_patience_min_delta)) or float(
            self.ensemble_patience_min_delta
        ) < 0.0:
            raise ValueError("ensemble_patience_min_delta must be finite and nonnegative.")
        evaluate_test_per_round = bool(
            test_data is not None and test_evaluation_policy == "per_round_diagnostic_holdout"
        )

        start_round = 1
        if resume_checkpoint:
            start_round = self._validate_restored_state(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                rng_seed=int(rng_seed),
                stop_target=stop_target,
            )
            if self.stopping_summary is not None:
                terminal = self.stopping_summary.stop_reason in {
                    "matched_stop_targets",
                    "fixed_train_margin_loss_reached",
                    "max_rounds_reached",
                    "planned_round_budget_reached",
                    "degenerate_weak_learner",
                    "ensemble_patience_reached",
                }
                if not terminal:
                    raise ValueError(
                        "Incompatible RealAdaBoost checkpoint: stopping summary is not terminal."
                    )
                return self

        cap_rounds = int(self.n_rounds)
        warmup_rounds = min(5, cap_rounds)
        effective_min_rounds = warmup_rounds
        planned_total_rounds = cap_rounds
        proxy_edge_lower_bound = math.inf
        use_planned_round_budget = False
        if stop_target is not None:
            # Honor the configured min_rounds as the authoritative stopping floor.
            # The 5-round warmup only stabilizes the budget estimate; it must not
            # silently raise a configured min_rounds < 5.
            effective_min_rounds = max(int(stop_target.min_rounds), 1)
            warmup_rounds = min(warmup_rounds, effective_min_rounds)
            use_planned_round_budget = bool(stop_target.use_planned_round_budget)

        use_jax = bool(_JAX_AVAILABLE)
        self._log(
            f"fit start cap={cap_rounds} train={train_data.n_samples} "
            f"validation={None if validation_data is None else validation_data.n_samples} "
            f"test={None if test_data is None else test_data.n_samples} "
            f"test_policy={test_evaluation_policy} "
            f"stop_target={stop_target is not None} use_jax={use_jax}"
        )

        train_y_np = np.asarray(train_data.y, dtype=np.float64)
        validation_y_np = None if validation_data is None else np.asarray(validation_data.y, dtype=np.float64)
        test_y_np = None if test_data is None else np.asarray(test_data.y, dtype=np.float64)

        if use_jax:
            assert jnp is not None
            train_y_backend = jnp.asarray(train_y_np, dtype=_JAX_FLOAT)
            validation_y_backend = (
                None if validation_y_np is None else jnp.asarray(validation_y_np, dtype=_JAX_FLOAT)
            )
            test_y_backend = None if test_y_np is None else jnp.asarray(test_y_np, dtype=_JAX_FLOAT)
            example_weights_t = jnp.full(
                (train_data.n_samples,),
                1.0 / train_data.n_samples,
                dtype=_JAX_FLOAT,
            )
            train_ensemble_score = jnp.zeros((train_data.n_samples,), dtype=_JAX_FLOAT)
            validation_monitor_ensemble_score = (
                None
                if validation_data is None
                else jnp.zeros((validation_data.n_samples,), dtype=_JAX_FLOAT)
            )
            validation_confirm_ensemble_score = (
                None
                if validation_data is None
                else jnp.zeros((validation_data.n_samples,), dtype=_JAX_FLOAT)
            )
            test_ensemble_score = (
                None if test_data is None else jnp.zeros((test_data.n_samples,), dtype=_JAX_FLOAT)
            )
            alpha_l1_t: Any = jnp.asarray(0.0, dtype=_JAX_FLOAT)
        else:
            train_y_backend = train_y_np
            validation_y_backend = validation_y_np
            test_y_backend = test_y_np
            example_weights_t = np.full(
                train_data.n_samples,
                1.0 / train_data.n_samples,
                dtype=np.float64,
            )
            train_ensemble_score = np.zeros(train_data.n_samples, dtype=np.float64)
            validation_monitor_ensemble_score = (
                None if validation_data is None else np.zeros(validation_data.n_samples, dtype=np.float64)
            )
            validation_confirm_ensemble_score = (
                None if validation_data is None else np.zeros(validation_data.n_samples, dtype=np.float64)
            )
            test_ensemble_score = (
                None if test_data is None else np.zeros(test_data.n_samples, dtype=np.float64)
            )
            alpha_l1_t = 0.0

        # Opt 1 (cont'd): upload the feature matrices to device once so per-round
        # in-graph prediction does not re-copy X host->device every round.
        if use_jax:
            assert jnp is not None
            train_X_for_predict: Any = jnp.asarray(train_data.X, dtype=_JAX_FLOAT)
            validation_X_for_predict = (
                None if validation_data is None else jnp.asarray(validation_data.X, dtype=_JAX_FLOAT)
            )
            test_X_for_predict = (
                None if test_data is None else jnp.asarray(test_data.X, dtype=_JAX_FLOAT)
            )
        else:
            train_X_for_predict = train_data.X
            validation_X_for_predict = None if validation_data is None else validation_data.X
            test_X_for_predict = None if test_data is None else test_data.X

        hit_streak = 0
        first_hit_round_index: Optional[int] = None
        first_hit_train_margin_loss: Optional[float] = None
        first_hit_validation_accuracy: Optional[float] = None
        first_hit_test_accuracy: Optional[float] = None
        deterministic_train_ensemble_score = np.zeros(train_data.n_samples, dtype=np.float64)
        ensemble_best_round_index: Optional[int] = None
        ensemble_best_train_margin_loss = math.inf
        ensemble_rounds_since_improvement = 0
        stop_reason = 'max_rounds_reached'

        if resume_checkpoint and self.resumable_state is not None:
            state = self.resumable_state

            def _backend_array(name: str, *, allow_none: bool = False) -> Any:
                raw = state.get(name)
                if raw is None and allow_none:
                    return None
                array = np.asarray(raw, dtype=np.float64)
                return jnp.asarray(array, dtype=_JAX_FLOAT) if use_jax else array

            example_weights_t = _backend_array("example_weights_after_update")
            train_ensemble_score = _backend_array("train_ensemble_score")
            validation_monitor_ensemble_score = _backend_array(
                "validation_monitor_ensemble_score", allow_none=True
            )
            validation_confirm_ensemble_score = _backend_array(
                "validation_confirm_ensemble_score", allow_none=True
            )
            test_ensemble_score = _backend_array("test_ensemble_score", allow_none=True)
            alpha_l1_value = float(state["alpha_l1_norm"])
            alpha_l1_t = jnp.asarray(alpha_l1_value, dtype=_JAX_FLOAT) if use_jax else alpha_l1_value
            hit_streak = int(state.get("hit_streak", 0))
            first_hit_round_index = state.get("first_hit_round_index")
            first_hit_train_margin_loss = state.get("first_hit_train_margin_loss")
            first_hit_validation_accuracy = state.get("first_hit_validation_accuracy")
            first_hit_test_accuracy = state.get("first_hit_test_accuracy")
            deterministic_train_ensemble_score = np.asarray(
                state.get(
                    "deterministic_train_ensemble_score",
                    np.zeros(train_data.n_samples, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            ensemble_best_round_index = state.get("ensemble_best_round_index")
            ensemble_best_train_margin_loss = float(
                state.get("ensemble_best_train_margin_loss", math.inf)
            )
            ensemble_rounds_since_improvement = int(
                state.get("ensemble_rounds_since_improvement", 0)
            )
            planned_total_rounds = int(state.get("planned_total_rounds", planned_total_rounds))
            proxy_edge_lower_bound = float(state.get("proxy_edge_lower_bound", proxy_edge_lower_bound))
            if state.get("pending_stop_reason") is not None:
                stop_reason = str(state["pending_stop_reason"])
                start_round = self.n_rounds + 1

        def _capture_resumable_state(
            completed_round: int,
            *,
            pending_stop_reason: Optional[str] = None,
        ) -> None:
            self.resumable_state = {
                "round_index": int(completed_round),
                "base_rng_seed": int(rng_seed),
                "completed_round_rng_stream_seeds": dict(
                    boosting_round_rng_seeds(int(rng_seed), int(completed_round))
                ),
                "next_round_rng_stream_seeds": (
                    None if int(completed_round) >= int(self.n_rounds)
                    else dict(boosting_round_rng_seeds(int(rng_seed), int(completed_round) + 1))
                ),
                "train_n_samples": train_data.n_samples,
                "validation_n_samples": None if validation_data is None else validation_data.n_samples,
                "test_n_samples": None if test_data is None else test_data.n_samples,
                "example_weights_after_update": np.asarray(example_weights_t, dtype=np.float64).tolist(),
                "train_ensemble_score": np.asarray(train_ensemble_score, dtype=np.float64).tolist(),
                "validation_monitor_ensemble_score": (
                    None if validation_monitor_ensemble_score is None
                    else np.asarray(validation_monitor_ensemble_score, dtype=np.float64).tolist()
                ),
                "validation_confirm_ensemble_score": (
                    None if validation_confirm_ensemble_score is None
                    else np.asarray(validation_confirm_ensemble_score, dtype=np.float64).tolist()
                ),
                "test_ensemble_score": (
                    None if test_ensemble_score is None
                    else np.asarray(test_ensemble_score, dtype=np.float64).tolist()
                ),
                "alpha_l1_norm": float(alpha_l1_t),
                "hit_streak": int(hit_streak),
                "first_hit_round_index": first_hit_round_index,
                "first_hit_train_margin_loss": first_hit_train_margin_loss,
                "first_hit_validation_accuracy": first_hit_validation_accuracy,
                "first_hit_test_accuracy": first_hit_test_accuracy,
                "deterministic_train_ensemble_score": deterministic_train_ensemble_score.tolist(),
                "ensemble_best_round_index": ensemble_best_round_index,
                "ensemble_best_train_margin_loss": float(ensemble_best_train_margin_loss),
                "ensemble_rounds_since_improvement": int(ensemble_rounds_since_improvement),
                "planned_total_rounds": int(planned_total_rounds),
                "proxy_edge_lower_bound": float(proxy_edge_lower_bound),
                "pending_stop_reason": pending_stop_reason,
                "stop_target": None if stop_target is None else asdict(stop_target),
            }

        for round_idx in range(start_round, self.n_rounds + 1):
            rng_stream_seeds = boosting_round_rng_seeds(int(rng_seed), round_idx)
            round_seed = int(rng_stream_seeds["round"])
            fit_rng = np.random.default_rng(rng_stream_seeds["fit"])
            edge_rng = np.random.default_rng(rng_stream_seeds["edge_score"])
            validation_monitor_rng = np.random.default_rng(rng_stream_seeds["validation_monitor"])
            validation_confirm_rng = np.random.default_rng(rng_stream_seeds["validation_confirm"])
            test_rng = np.random.default_rng(rng_stream_seeds["test_score"])
            learner = weak_learner_factory()

            start_time = time.perf_counter()
            if self._should_log_round(round_idx):
                self._log(
                    f"round {round_idx}/{cap_rounds} start "
                    f"planned_total={planned_total_rounds} seed={round_seed}"
                )
            fit_weights = _example_weights_for_fit(learner, example_weights_t, use_jax=use_jax)
            fit_t0 = time.perf_counter()
            fit_result = learner.fit_weighted(train_data, fit_weights, fit_rng)
            fit_accounting = _learner_shot_accounting(learner)
            if self._should_log_round(round_idx):
                self._log(
                    f"round {round_idx}/{cap_rounds} weak learner fit done "
                    f"epochs={fit_result.epochs_run} converged={fit_result.converged} "
                    f"surrogate_loss={fit_result.weighted_surrogate_loss_t:.6g} "
                    f"elapsed={time.perf_counter() - fit_t0:.2f}s"
                )

            # Use the weak learner's own training smoothing for the surrogate
            # certificate when exposed, so the certificate reflects the smoothing
            # actually used in training. Falls back to the booster default.
            round_surrogate_sigma = float(getattr(learner, "sigma", self.surrogate_sigma))

            before_edge_accounting = _learner_shot_accounting(learner)
            with _training_phase(learner):
                train_scores_raw_backend = _weak_scores_backend(
                    learner, train_X_for_predict, edge_rng, use_jax=use_jax,
                    prefer_jax=self.use_in_graph_prediction,
                )
            edge_accounting = _shot_accounting_delta(
                _learner_shot_accounting(learner),
                before_edge_accounting,
                source="edge",
            )

            if use_jax:
                assert jnp is not None
                example_weights_before_backend = example_weights_t
                (
                    edge_arr,
                    weak_sign_arr,
                    train_scores_backend,
                    boost_coef_arr,
                    example_weights_t,
                    train_ensemble_score,
                    alpha_l1_t,
                    train_error_arr,
                    train_margin_loss_arr,
                    surrogate_smooth_hinge_loss_arr,
                    surrogate_edge_lower_bound_arr,
                ) = _jax_train_round(
                    example_weights_t,
                    train_y_backend,
                    jnp.asarray(train_scores_raw_backend, dtype=_JAX_FLOAT),
                    train_ensemble_score,
                    alpha_l1_t,
                    float(self.margin_threshold),
                    float(self.clip_edge),
                    round_surrogate_sigma,
                )
            else:
                train_scores_raw_np = np.asarray(train_scores_raw_backend, dtype=np.float64)
                train_scores_raw = _np_clip_scores(train_scores_raw_np)
                edge_t, weak_sign, train_scores_backend, boost_coef_t = _np_edge_and_boost(
                    example_weights_t,
                    train_y_backend,
                    train_scores_raw,
                    self.clip_edge,
                )
                example_weights_before = np.asarray(example_weights_t, dtype=np.float64)
                example_weights_t = _np_update_weights(
                    example_weights_t,
                    train_y_backend,
                    train_scores_backend,
                    boost_coef_t,
                )
                train_ensemble_score = _np_update_ensemble(
                    train_ensemble_score,
                    train_scores_backend,
                    boost_coef_t,
                )
                alpha_l1_t = float(alpha_l1_t) + abs(boost_coef_t)
                train_error, _, train_margin_loss = _np_dataset_metrics(
                    train_y_backend,
                    train_ensemble_score,
                    self.margin_threshold,
                    alpha_l1_t,
                )
                train_scores_np = np.asarray(train_scores_backend, dtype=np.float64)
                train_ensemble_score_np = np.asarray(train_ensemble_score, dtype=np.float64)
                surrogate_smooth_hinge_loss_t = _np_smooth_hinge_loss_from_scores(
                    example_weights_before,
                    train_y_backend,
                    train_scores_backend,
                    sigma=round_surrogate_sigma,
                )
                surrogate_edge_lower_bound_t = float(1.0 - surrogate_smooth_hinge_loss_t)

            validation_scores_np = None
            validation_accuracy = None
            validation_error = None
            validation_margin_loss = None
            validation_monitor_accuracy = None
            validation_monitor_error = None
            validation_monitor_margin_loss = None
            validation_monitor_triggered = False
            validation_confirmation_attempted = False
            validation_confirmation_policy = None
            validation_monitor_accounting = empty_shot_accounting_dict(source="validation_monitor")
            validation_confirm_accounting = empty_shot_accounting_dict(source="validation_confirm")
            external_confirmation_accounting = empty_shot_accounting_dict(
                source="prior_learner_validation_confirm"
            )
            if validation_data is not None:
                validation_confirmation_policy = str(self.validation_confirmation_policy)
                sign_scalar = float(weak_sign_arr) if use_jax else float(weak_sign)
                coef_scalar = float(boost_coef_arr) if use_jax else float(boost_coef_t)
                before_validation_monitor_accounting = _learner_shot_accounting(learner)
                with _training_phase(learner):
                    validation_monitor_scores_raw_backend = _weak_scores_backend(
                        learner, validation_X_for_predict, validation_monitor_rng, use_jax=use_jax,
                        prefer_jax=self.use_in_graph_prediction,
                    )
                validation_monitor_accounting = _shot_accounting_delta(
                    _learner_shot_accounting(learner),
                    before_validation_monitor_accounting,
                    source="validation_monitor",
                )
                validation_monitor_scores_backend = sign_scalar * _np_clip_scores(
                    np.asarray(validation_monitor_scores_raw_backend, dtype=np.float64)
                )
                assert validation_monitor_ensemble_score is not None
                validation_monitor_ensemble_score = _np_update_ensemble(
                    validation_monitor_ensemble_score,
                    validation_monitor_scores_backend,
                    coef_scalar,
                )
                (
                    validation_monitor_error,
                    validation_monitor_accuracy,
                    validation_monitor_margin_loss,
                ) = _np_dataset_metrics(
                    validation_y_np,
                    validation_monitor_ensemble_score,
                    self.margin_threshold,
                    float(alpha_l1_t),
                )
                validation_monitor_triggered = bool(
                    stop_target is not None
                    and not fixed_train_margin_stop
                    and validation_monitor_accuracy is not None
                    and validation_monitor_accuracy + stop_target.validation_accuracy_tolerance
                    >= stop_target.target_validation_accuracy
                )
                should_confirm = bool(
                    validation_confirmation_policy == "every_round_independent_stream"
                    or validation_monitor_triggered
                )
                if should_confirm:
                    validation_confirmation_attempted = True
                    if validation_confirmation_policy == "every_round_independent_stream":
                        before = _learner_shot_accounting(learner)
                        with _training_phase(learner):
                            confirm_raw = learner.predict_score(
                                validation_data.X, rng=validation_confirm_rng
                            )
                        validation_confirm_accounting = _shot_accounting_delta(
                            _learner_shot_accounting(learner), before,
                            source="validation_confirm",
                        )
                        validation_confirm_scores_backend = sign_scalar * _np_clip_scores(confirm_raw)
                        assert validation_confirm_ensemble_score is not None
                        validation_confirm_ensemble_score = _np_update_ensemble(
                            validation_confirm_ensemble_score,
                            validation_confirm_scores_backend,
                            coef_scalar,
                        )
                    else:
                        # A confirmation is a fresh evaluation of the entire
                        # accepted prefix plus the current candidate. Replaying
                        # the prefix avoids spending confirmation shots on monitor
                        # failures while preserving an independent ensemble stream.
                        validation_confirm_ensemble_score = np.zeros(
                            validation_data.n_samples, dtype=np.float64
                        )
                        deltas: list[dict[str, Any]] = []
                        prior_deltas: list[dict[str, Any]] = []
                        learners = [*self.weak_learners, learner]
                        signs = [*self.weak_learner_signs, sign_scalar]
                        coefs = [*self.boost_coefs, coef_scalar]
                        for learner_idx, (confirm_learner, confirm_sign, confirm_coef) in enumerate(
                            zip(learners, signs, coefs)
                        ):
                            before = _learner_shot_accounting(confirm_learner)
                            with _training_phase(confirm_learner, audit_role="confirm"):
                                raw = confirm_learner.predict_score(
                                    validation_data.X,
                                    rng=np.random.default_rng(
                                        stable_int_seed(
                                            "real_adaboost_validation_confirmation",
                                            int(rng_seed), int(round_idx), int(learner_idx),
                                        )
                                    ),
                                )
                            confirm_delta = _shot_accounting_delta(
                                _learner_shot_accounting(confirm_learner), before,
                                source="validation_confirm",
                            )
                            deltas.append(confirm_delta)
                            if learner_idx < len(learners) - 1:
                                prior_deltas.append(confirm_delta)
                            signed = float(confirm_sign) * _np_clip_scores(raw)
                            validation_confirm_ensemble_score = _np_update_ensemble(
                                validation_confirm_ensemble_score, signed, float(confirm_coef)
                            )
                            if learner_idx == len(learners) - 1:
                                validation_confirm_scores_backend = signed
                        validation_confirm_accounting = sum_shot_accounting_dicts(deltas)
                        external_confirmation_accounting = sum_shot_accounting_dicts(
                            prior_deltas
                        )
                    validation_error, validation_accuracy, validation_margin_loss = _np_dataset_metrics(
                        validation_y_np,
                        validation_confirm_ensemble_score,
                        self.margin_threshold,
                        float(alpha_l1_t),
                    )
                    validation_scores_np = np.asarray(
                        validation_confirm_scores_backend, dtype=np.float64
                    )

            test_scores_np = None
            test_accuracy = None
            test_margin_loss = None
            diagnostic_test_accounting = empty_shot_accounting_dict(source="diagnostic_test")
            if evaluate_test_per_round:
                before_diagnostic_test_accounting = _learner_shot_accounting(learner)
                with _training_phase(learner):
                    test_scores_raw_backend = _weak_scores_backend(
                        learner, test_X_for_predict, test_rng, use_jax=use_jax,
                        prefer_jax=self.use_in_graph_prediction,
                    )
                diagnostic_test_accounting = _shot_accounting_delta(
                    _learner_shot_accounting(learner),
                    before_diagnostic_test_accounting,
                    source="diagnostic_test",
                )
                if use_jax:
                    assert jnp is not None
                    test_scores_backend = weak_sign_arr * _jax_clip_scores(
                        jnp.asarray(test_scores_raw_backend, dtype=_JAX_FLOAT)
                    )
                    assert test_ensemble_score is not None
                    assert test_y_backend is not None
                    test_ensemble_score = _jax_update_ensemble(
                        test_ensemble_score,
                        test_scores_backend,
                        boost_coef_arr,
                    )
                    test_error_arr, test_accuracy_arr, test_margin_loss_arr = _jax_dataset_metrics(
                        test_y_backend,
                        test_ensemble_score,
                        float(self.margin_threshold),
                        alpha_l1_t,
                    )
                    del test_error_arr
                    test_scores_np = np.asarray(test_scores_backend, dtype=np.float64)
                    test_accuracy = float(test_accuracy_arr)
                    test_margin_loss = float(test_margin_loss_arr)
                else:
                    test_scores_raw_np = np.asarray(test_scores_raw_backend, dtype=np.float64)
                    test_scores_backend = weak_sign * _np_clip_scores(test_scores_raw_np)
                    assert test_ensemble_score is not None
                    assert test_y_backend is not None
                    test_ensemble_score = _np_update_ensemble(
                        test_ensemble_score,
                        test_scores_backend,
                        boost_coef_t,
                    )
                    _, test_accuracy, test_margin_loss = _np_dataset_metrics(
                        test_y_backend,
                        test_ensemble_score,
                        self.margin_threshold,
                        alpha_l1_t,
                    )
                    test_scores_np = np.asarray(test_scores_backend, dtype=np.float64)

            if use_jax:
                edge_t = float(edge_arr)
                weak_sign = float(weak_sign_arr)
                boost_coef_t = float(boost_coef_arr)
                train_error = float(train_error_arr)
                train_margin_loss = float(train_margin_loss_arr)
                surrogate_smooth_hinge_loss_t = float(surrogate_smooth_hinge_loss_arr)
                surrogate_edge_lower_bound_t = float(surrogate_edge_lower_bound_arr)
                example_weights_before = np.asarray(example_weights_before_backend, dtype=np.float64)
                train_scores_np = np.asarray(train_scores_backend, dtype=np.float64)
                train_ensemble_score_np = np.asarray(train_ensemble_score, dtype=np.float64)

            round_train_margin_mu = (
                float(stop_target.train_margin_mu)
                if fixed_train_margin_stop and stop_target is not None
                else float(self.margin_threshold)
            )
            train_margin_loss = normalized_empirical_ramp_loss(
                train_y_np,
                train_ensemble_score_np,
                float(alpha_l1_t),
                round_train_margin_mu,
            )
            round_train_margin_target = (
                float(stop_target.train_margin_target)
                if fixed_train_margin_stop
                and stop_target is not None
                and stop_target.train_margin_target is not None
                else None
            )
            round_train_margin_target_reached = (
                None
                if round_train_margin_target is None
                else bool(train_margin_loss <= round_train_margin_target)
            )

            train_margin_candidate_reached = round_train_margin_target_reached
            train_margin_confirmation_attempted = False
            train_margin_confirmation_loss = None
            train_margin_confirmation_reached = None
            train_margin_confirmation_ensemble_score = None
            train_margin_confirmation_accounting = empty_shot_accounting_dict(
                source="train_margin_confirm"
            )
            external_target_confirmation_accounting = empty_shot_accounting_dict(
                source="prior_learner_train_margin_confirm"
            )
            if (
                fixed_train_margin_stop
                and self.train_margin_confirmation_policy
                == "monitor_then_confirm_on_success_independent_stream"
                and bool(train_margin_candidate_reached)
            ):
                train_margin_confirmation_attempted = True
                confirm_ensemble = np.zeros(train_data.n_samples, dtype=np.float64)
                confirm_deltas: list[dict[str, Any]] = []
                prior_confirm_deltas: list[dict[str, Any]] = []
                learners = [*self.weak_learners, learner]
                signs = [*self.weak_learner_signs, float(weak_sign)]
                coefs = [*self.boost_coefs, float(boost_coef_t)]
                if hasattr(self, "audit_confirmation_group"):
                    self.audit_confirmation_group(len(learners), train_data.n_samples)
                for learner_idx, (confirm_learner, confirm_sign, confirm_coef) in enumerate(
                    zip(learners, signs, coefs)
                ):
                    before = _learner_shot_accounting(confirm_learner)
                    with _training_phase(confirm_learner, audit_role="confirm"):
                        raw = confirm_learner.predict_score(
                            train_data.X,
                            rng=np.random.default_rng(
                                stable_int_seed(
                                    "real_adaboost_train_margin_confirmation",
                                    int(rng_seed),
                                    int(round_idx),
                                    int(learner_idx),
                                )
                            ),
                        )
                    delta = _shot_accounting_delta(
                        _learner_shot_accounting(confirm_learner),
                        before,
                        source="train_margin_confirm",
                    )
                    confirm_deltas.append(delta)
                    if learner_idx < len(learners) - 1:
                        prior_confirm_deltas.append(delta)
                    signed = float(confirm_sign) * _np_clip_scores(raw)
                    confirm_ensemble = _np_update_ensemble(
                        confirm_ensemble, signed, float(confirm_coef)
                    )
                train_margin_confirmation_accounting = sum_shot_accounting_dicts(
                    confirm_deltas
                )
                external_target_confirmation_accounting = sum_shot_accounting_dicts(
                    prior_confirm_deltas
                )
                train_margin_confirmation_ensemble_score = np.asarray(
                    confirm_ensemble, dtype=np.float64
                )
                train_margin_confirmation_loss = normalized_empirical_ramp_loss(
                    train_y_np,
                    train_margin_confirmation_ensemble_score,
                    float(alpha_l1_t),
                    float(round_train_margin_mu),
                )
                train_margin_confirmation_reached = bool(
                    train_margin_confirmation_loss <= float(round_train_margin_target)
                )
                round_train_margin_target_reached = train_margin_confirmation_reached

            total_round_accounting = sum_shot_accounting_dicts(
                [
                    _learner_shot_accounting(learner),
                    external_confirmation_accounting,
                    external_target_confirmation_accounting,
                ]
            )
            shot_accounting = _annotate_boosting_component_accounting(
                total_round_accounting,
                fit_accounting=fit_accounting,
                edge_accounting=edge_accounting,
                validation_monitor_accounting=validation_monitor_accounting,
                validation_confirm_accounting=validation_confirm_accounting,
                train_margin_confirm_accounting=train_margin_confirmation_accounting,
                diagnostic_test_accounting=diagnostic_test_accounting,
            )

            round_result = BoostingRoundResult(
                round_index=round_idx,
                edge_t=edge_t,
                boost_coef_t=boost_coef_t,
                train_scores=train_scores_np.copy(),
                train_ensemble_score=train_ensemble_score_np.copy(),
                example_weights_before_update=np.asarray(example_weights_before, dtype=np.float64).copy(),
                example_weights_after_update=np.asarray(example_weights_t, dtype=np.float64).copy(),
                empirical_train_error=train_error,
                empirical_train_margin_loss=train_margin_loss,
                weighted_surrogate_loss_t=fit_result.weighted_surrogate_loss_t,
                surrogate_smooth_hinge_loss_t=surrogate_smooth_hinge_loss_t,
                surrogate_edge_lower_bound_t=surrogate_edge_lower_bound_t,
                surrogate_sigma_t=round_surrogate_sigma,
                validation_scores=None if validation_scores_np is None else validation_scores_np.copy(),
                validation_ensemble_score=(
                    None
                    if validation_monitor_ensemble_score is None
                    or self.train_margin_confirmation_policy == "none"
                    else np.asarray(
                        validation_monitor_ensemble_score, dtype=np.float64
                    ).copy()
                ),
                validation_accuracy=validation_accuracy,
                empirical_validation_error=validation_error,
                empirical_validation_margin_loss=validation_margin_loss,
                validation_monitor_accuracy=validation_monitor_accuracy,
                empirical_validation_monitor_error=validation_monitor_error,
                empirical_validation_monitor_margin_loss=validation_monitor_margin_loss,
                validation_monitor_triggered=validation_monitor_triggered,
                validation_confirmation_attempted=validation_confirmation_attempted,
                validation_confirmation_policy=validation_confirmation_policy,
                test_scores=None if test_scores_np is None else test_scores_np.copy(),
                test_accuracy=test_accuracy,
                empirical_test_margin_loss=test_margin_loss,
                test_evaluation_policy=(test_evaluation_policy if test_data is not None else None),
                test_evaluation_is_final=False,
                test_evaluation_is_diagnostic=bool(evaluate_test_per_round),
                shot_budget_consumed=_shot_budget_consumed(shot_accounting),
                shot_accounting=shot_accounting,
                wall_clock_seconds=float(time.perf_counter() - start_time),
                weak_learner_epochs_run=fit_result.epochs_run,
                weak_learner_converged=fit_result.converged,
                weak_learner_convergence_diagnostics=(
                    None if fit_result.convergence_diagnostics is None else dict(fit_result.convergence_diagnostics)
                ),
                rng_seed=round_seed,
                rng_stream_seeds=dict(rng_stream_seeds),
                train_margin_mu=(round_train_margin_mu if fixed_train_margin_stop else None),
                train_margin_target=round_train_margin_target,
                train_margin_target_reached=round_train_margin_target_reached,
                train_margin_candidate_reached=train_margin_candidate_reached,
                train_margin_confirmation_attempted=train_margin_confirmation_attempted,
                train_margin_confirmation_policy=(
                    self.train_margin_confirmation_policy
                    if fixed_train_margin_stop
                    else None
                ),
                train_margin_confirmation_loss=train_margin_confirmation_loss,
                train_margin_confirmation_reached=train_margin_confirmation_reached,
                train_margin_confirmation_ensemble_score=(
                    None
                    if train_margin_confirmation_ensemble_score is None
                    else train_margin_confirmation_ensemble_score.copy()
                ),
                train_margin_confirmation_shot_accounting=(
                    train_margin_confirmation_accounting
                    if train_margin_confirmation_attempted
                    else None
                ),
            )

            self.weak_learners.append(learner)
            self.weak_learner_signs.append(weak_sign)
            self.boost_coefs.append(boost_coef_t)
            self.round_results.append(round_result)

            if self.ensemble_patience_rounds is not None:
                deterministic_weak_scores = float(weak_sign) * _deterministic_mean_scores(
                    learner, train_data.X
                )
                deterministic_train_ensemble_score = _np_update_ensemble(
                    deterministic_train_ensemble_score,
                    deterministic_weak_scores,
                    float(boost_coef_t),
                )
                deterministic_loss = normalized_empirical_ramp_loss(
                    train_y_np,
                    deterministic_train_ensemble_score,
                    float(alpha_l1_t),
                    float(round_train_margin_mu),
                )
                improved = bool(
                    ensemble_best_round_index is None
                    or deterministic_loss
                    < ensemble_best_train_margin_loss
                    - float(self.ensemble_patience_min_delta)
                )
                if improved:
                    ensemble_best_round_index = int(round_idx)
                    ensemble_best_train_margin_loss = float(deterministic_loss)
                    ensemble_rounds_since_improvement = 0
                else:
                    ensemble_rounds_since_improvement += 1
                round_result.deterministic_train_margin_loss = float(deterministic_loss)
                round_result.ensemble_patience_improved = improved
                round_result.ensemble_patience_rounds_since_improvement = int(
                    ensemble_rounds_since_improvement
                )

            proxy_edge_lower_bound = min(proxy_edge_lower_bound, float(edge_t))
            if stop_target is not None and use_planned_round_budget:
                planned_total_rounds = _planned_round_budget_real(
                    edge_lower_bound=proxy_edge_lower_bound,
                    margin_threshold=self.margin_threshold,
                    target_margin_loss=stop_target.target_margin_loss,
                    cap_rounds=cap_rounds,
                    warmup_rounds=warmup_rounds,
                )

            if self._should_log_round(round_idx):
                budget_suffix = (
                    ""
                    if stop_target is None
                    else _round_budget_diagnostic_suffix(
                        round_results=self.round_results,
                        y_train=train_y_np,
                        margin_threshold=self.margin_threshold,
                        target_margin_loss=stop_target.target_margin_loss,
                    )
                )
                self._log(
                    f"round {round_idx}/{cap_rounds} done "
                    f"edge={edge_t:.6g} alpha={boost_coef_t:.6g} "
                    f"train_margin_loss={train_margin_loss:.6g} "
                    f"val_monitor_acc={validation_monitor_accuracy} "
                    f"val_confirm_acc={validation_accuracy} test_acc={test_accuracy} "
                    f"planned_total={planned_total_rounds} hit_streak={hit_streak} "
                    f"elapsed={round_result.wall_clock_seconds:.2f}s" + budget_suffix
                )

            if stop_target is not None:
                if fixed_train_margin_stop:
                    criteria_hit = (
                        bool(round_result.train_margin_confirmation_reached)
                        if self.train_margin_confirmation_policy != "none"
                        else fixed_train_margin_loss_decision(
                            round_result,
                            stop_target,
                            round_index=round_idx,
                        )
                    )
                else:
                    assert stop_target.target_validation_accuracy is not None
                    meets_validation_target = (
                        validation_accuracy is not None
                        and validation_accuracy + stop_target.validation_accuracy_tolerance
                        >= stop_target.target_validation_accuracy
                    )
                    meets_validation_monitor_target = (
                        validation_monitor_accuracy is not None
                        and validation_monitor_accuracy + stop_target.validation_accuracy_tolerance
                        >= stop_target.target_validation_accuracy
                    )
                    criteria_hit = (
                        round_idx >= effective_min_rounds
                        and meets_validation_monitor_target
                        and meets_validation_target
                    )
                if criteria_hit:
                    hit_streak += 1
                    if first_hit_round_index is None:
                        first_hit_round_index = round_idx
                        first_hit_train_margin_loss = round_result.empirical_train_margin_loss
                        first_hit_validation_accuracy = validation_accuracy
                        first_hit_test_accuracy = round_result.test_accuracy
                else:
                    hit_streak = 0

                if hit_streak >= stop_target.patience:
                    stop_reason = (
                        'fixed_train_margin_loss_reached'
                        if fixed_train_margin_stop
                        else 'matched_stop_targets'
                    )
                    self._log(
                        f"stopping at round {round_idx}: matched stop targets "
                        f"hit_streak={hit_streak}/{stop_target.patience}"
                    )
                    _capture_resumable_state(round_idx, pending_stop_reason=stop_reason)
                    self._maybe_write_checkpoint(round_idx, reason=stop_reason)
                    break

            if (
                self.ensemble_patience_rounds is not None
                and ensemble_rounds_since_improvement >= int(self.ensemble_patience_rounds)
            ):
                stop_reason = "ensemble_patience_reached"
                self._log(
                    f"stopping at round {round_idx}: deterministic ensemble ramp "
                    f"did not improve by > {self.ensemble_patience_min_delta:.6g} for "
                    f"{self.ensemble_patience_rounds} accepted rounds"
                )
                _capture_resumable_state(round_idx, pending_stop_reason=stop_reason)
                self._maybe_write_checkpoint(round_idx, reason=stop_reason)
                break

            if (
                stop_target is not None
                and use_planned_round_budget
                and planned_total_rounds < cap_rounds
                and round_idx >= planned_total_rounds
            ):
                stop_reason = 'planned_round_budget_reached'
                self._log(
                    f"stopping at round {round_idx}: planned round budget reached "
                    f"planned_total={planned_total_rounds} cap={cap_rounds}"
                )
                _capture_resumable_state(round_idx, pending_stop_reason=stop_reason)
                self._maybe_write_checkpoint(round_idx, reason=stop_reason)
                break

            _capture_resumable_state(round_idx)
            self._maybe_write_checkpoint(round_idx, reason='round_completed')

        if (
            test_data is not None
            and test_evaluation_policy == "final_only"
            and self.round_results
            and (
                self.train_margin_confirmation_policy == "none"
                or stop_reason == "fixed_train_margin_loss_reached"
                or (
                    self.ensemble_patience_rounds is not None
                    and stop_reason in {"ensemble_patience_reached", "max_rounds_reached"}
                )
            )
        ):
            final_test_t0 = time.perf_counter()
            selected_round_index = (
                int(first_hit_round_index)
                if stop_reason in {"matched_stop_targets", "fixed_train_margin_loss_reached"}
                and first_hit_round_index is not None
                else int(
                    ensemble_best_round_index
                    if self.ensemble_patience_rounds is not None
                    and ensemble_best_round_index is not None
                    and stop_reason in {"ensemble_patience_reached", "max_rounds_reached"}
                    else len(self.round_results)
                )
            )
            selection_reason = (
                (
                    "first_train_margin_hit"
                    if fixed_train_margin_stop
                    else "first_confirmed_hit"
                )
                if stop_reason in {"matched_stop_targets", "fixed_train_margin_loss_reached"}
                and first_hit_round_index is not None
                else (
                    "best_deterministic_train_ramp"
                    if self.ensemble_patience_rounds is not None
                    and stop_reason in {"ensemble_patience_reached", "max_rounds_reached"}
                    else "terminal_budget"
                )
            )
            final_test_ensemble_score_np = np.zeros(test_data.n_samples, dtype=np.float64)
            final_test_accounting_deltas: list[dict[str, Any]] = []
            for learner_idx, (learner, weak_sign_t, boost_coef_t) in enumerate(
                zip(
                    self.weak_learners[:selected_round_index],
                    self.weak_learner_signs[:selected_round_index],
                    self.boost_coefs[:selected_round_index],
                ),
                start=1,
            ):
                before_accounting = _learner_shot_accounting(learner)
                learner_test_seed = boosting_round_rng_seeds(int(rng_seed), learner_idx)["test_score"]
                test_scores_raw_backend = _weak_scores_backend(
                    learner,
                    test_X_for_predict,
                    np.random.default_rng(learner_test_seed),
                    use_jax=use_jax,
                    prefer_jax=self.use_in_graph_prediction,
                )
                test_scores_np = float(weak_sign_t) * _np_clip_scores(
                    np.asarray(test_scores_raw_backend, dtype=np.float64)
                )
                final_test_ensemble_score_np = _np_update_ensemble(
                    final_test_ensemble_score_np,
                    test_scores_np,
                    float(boost_coef_t),
                )
                after_accounting = _learner_shot_accounting(learner)
                final_test_accounting_deltas.append(
                    _shot_accounting_delta(
                        after_accounting,
                        before_accounting,
                        source="final_test",
                    )
                )
            alpha_l1_final = float(
                np.sum(np.abs(np.asarray(self.boost_coefs[:selected_round_index], dtype=np.float64)))
            )
            assert test_y_np is not None
            _, final_test_accuracy, final_test_margin_loss = _np_dataset_metrics(
                test_y_np,
                final_test_ensemble_score_np,
                self.margin_threshold,
                alpha_l1_final,
            )
            final_test_accounting = sum_shot_accounting_dicts(final_test_accounting_deltas)
            selected_round = self.round_results[selected_round_index - 1]
            selected_round.test_accuracy = float(final_test_accuracy)
            selected_round.empirical_test_margin_loss = float(final_test_margin_loss)
            selected_round.test_scores = final_test_ensemble_score_np.copy()
            selected_round.test_evaluation_policy = "final_only"
            selected_round.test_evaluation_is_final = True
            selected_round.test_evaluation_is_diagnostic = False
            selected_round.final_test_selection_reason = selection_reason
            selected_round.final_test_shot_accounting = final_test_accounting
            selected_round.final_test_wall_clock_seconds = float(time.perf_counter() - final_test_t0)
            if selection_reason in {"first_confirmed_hit", "first_train_margin_hit"}:
                first_hit_test_accuracy = float(final_test_accuracy)
            self._log(
                f"final-only test evaluation done round={selected_round.round_index} "
                f"test_acc={final_test_accuracy} test_margin_loss={final_test_margin_loss}"
            )

        self._log(f"fit complete rounds={len(self.round_results)} stop_reason={stop_reason}")

        if stop_target is not None:
            self.stopping_summary = BoostingStopSummary(
                target_reached=stop_reason in {
                    'matched_stop_targets',
                    'fixed_train_margin_loss_reached',
                },
                stop_round_index=len(self.round_results),
                stop_reason=stop_reason,
                first_hit_round_index=first_hit_round_index,
                target_margin_loss=float(stop_target.target_margin_loss),
                target_validation_accuracy=(
                    None
                    if stop_target.target_validation_accuracy is None
                    else float(stop_target.target_validation_accuracy)
                ),
                validation_accuracy_tolerance=float(stop_target.validation_accuracy_tolerance),
                min_rounds=int(stop_target.min_rounds),
                patience=int(stop_target.patience),
                hit_train_margin_loss=first_hit_train_margin_loss,
                hit_validation_accuracy=first_hit_validation_accuracy,
                hit_test_accuracy=first_hit_test_accuracy,
                stop_mode=str(stop_target.stop_mode),
                train_margin_mu=stop_target.train_margin_mu,
                train_margin_target=stop_target.train_margin_target,
                ensemble_patience_rounds=self.ensemble_patience_rounds,
                ensemble_patience_min_delta=(
                    float(self.ensemble_patience_min_delta)
                    if self.ensemble_patience_rounds is not None
                    else None
                ),
                ensemble_best_round_index=ensemble_best_round_index,
                ensemble_best_train_margin_loss=(
                    None
                    if ensemble_best_round_index is None
                    else float(ensemble_best_train_margin_loss)
                ),
                ensemble_rounds_since_improvement=(
                    int(ensemble_rounds_since_improvement)
                    if self.ensemble_patience_rounds is not None
                    else None
                ),
            )

        self._maybe_write_checkpoint(len(self.round_results), reason=stop_reason, force=True)
        return self

    def predict_score(
        self,
        x: Array,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        del rng
        if not self.weak_learners:
            return np.zeros(shape=(x.shape[0],), dtype=np.float64)

        x = np.asarray(x, dtype=np.float64)
        ensemble_score_np = np.zeros(shape=(x.shape[0],), dtype=np.float64)
        for learner, sign_t, boost_coef_t in zip(self.weak_learners, self.weak_learner_signs, self.boost_coefs):
            # Clip to [-1, 1] to match the training-time ensemble construction,
            # which clips weak scores before accumulation.
            weak_scores = np.clip(np.asarray(learner.predict_score(x), dtype=np.float64), -1.0, 1.0)
            ensemble_score_np = ensemble_score_np + boost_coef_t * sign_t * weak_scores
        return ensemble_score_np

    def predict_label(
        self,
        x: Array,
        threshold: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        score = self.predict_score(x, rng=rng)
        return np.where(score >= threshold, 1, -1).astype(np.int8)

    def margin(self, dataset: BinaryDataset) -> Array:
        return dataset.y * self.predict_score(dataset.X)

    def normalized_margin(self, dataset: BinaryDataset) -> Array:
        alpha_l1 = float(np.sum(np.abs(np.asarray(self.boost_coefs, dtype=np.float64))))
        return normalized_margin(dataset.y, self.predict_score(dataset.X), alpha_l1_norm=alpha_l1)

    def empirical_margin_loss(self, dataset: BinaryDataset, margin_threshold: Optional[float] = None) -> float:
        threshold = self.margin_threshold if margin_threshold is None else margin_threshold
        return empirical_margin_loss(self.normalized_margin(dataset), threshold)

    def resource_summary(self, round_limit: Optional[int] = None) -> dict:
        rows = self.round_results if round_limit is None else self.round_results[: int(round_limit)]
        accounting = sum_shot_accounting_dicts(result.shot_accounting for result in rows)
        final_test_accounting = sum_shot_accounting_dicts(
            result.final_test_shot_accounting for result in rows
        )
        reported_total_accounting = sum_shot_accounting_dicts([accounting, final_test_accounting])
        total_weak_learner_epochs = sum(
            int(result.weak_learner_epochs_run or 0)
            for result in rows
        )
        return {
            'algorithm': 'real_adaboost',
            'n_rounds': len(rows),
            'total_weak_learner_epochs': int(total_weak_learner_epochs),
            'weak_learner_converged_count': sum(
                result.weak_learner_converged is True
                for result in rows
            ),
            'weak_learner_censored_count': sum(
                result.weak_learner_converged is False
                for result in rows
            ),
            'primary_resource_basis': 'validation_stop_excludes_final_test',
            'train_effective_shot_budget': accounting['train_effective_shots'],
            'inference_effective_shot_budget': accounting['inference_effective_shots'],
            'total_effective_shot_budget': accounting['total_effective_shots'],
            'train_raw_shot_budget': accounting['train_raw_shots'],
            'inference_raw_shot_budget': accounting['inference_raw_shots'],
            'total_raw_shot_budget': accounting['total_raw_shots'],
            'train_shot_budget': accounting['train_effective_shots'],
            'inference_shot_budget': accounting['inference_effective_shots'],
            'total_shot_budget': accounting['total_effective_shots'],
            'train_model_evaluations': accounting['train_model_evaluations'],
            'inference_model_evaluations': accounting['inference_model_evaluations'],
            'total_model_evaluations': accounting['total_model_evaluations'],
            'train_trajectory_evaluations': accounting['train_trajectory_evaluations'],
            'inference_trajectory_evaluations': accounting['inference_trajectory_evaluations'],
            'total_trajectory_evaluations': accounting['total_trajectory_evaluations'],
            'trajectory_multiplier_counts_as_shots': accounting.get('trajectory_multiplier_counts_as_shots'),
            'shots_per_expectation': accounting.get('shots_per_expectation'),
            'measurement_sampling_distribution': accounting.get('measurement_sampling_distribution'),
            'measurement_gradient_mode': accounting.get('measurement_gradient_mode'),
            'gradient_loss_estimator': accounting.get('gradient_loss_estimator'),
            'gradient_unshifted_shots': accounting.get('gradient_unshifted_shots'),
            'gradient_shifted_shots': accounting.get('gradient_shifted_shots'),
            'shifted_parameter_count': accounting.get('shifted_parameter_count'),
            'shifted_gate_occurrence_count': accounting.get('shifted_gate_occurrence_count'),
            'unshifted_loss_evaluations': accounting.get('unshifted_loss_evaluations'),
            'shifted_gradient_evaluations': accounting.get('shifted_gradient_evaluations'),
            'training_monitor_evaluations': accounting.get('training_monitor_evaluations'),
            'boosting_component_ledger_basis': accounting.get('boosting_component_ledger_basis'),
            'component_train_model_evaluations': accounting.get('component_train_model_evaluations'),
            'component_gradient_model_evaluations': accounting.get('component_gradient_model_evaluations'),
            'component_training_monitor_model_evaluations': accounting.get(
                'component_training_monitor_model_evaluations'
            ),
            'component_edge_model_evaluations': accounting.get('component_edge_model_evaluations'),
            'component_validation_monitor_model_evaluations': accounting.get(
                'component_validation_monitor_model_evaluations'
            ),
            'component_validation_confirm_model_evaluations': accounting.get(
                'component_validation_confirm_model_evaluations'
            ),
            'component_diagnostic_test_model_evaluations': accounting.get(
                'component_diagnostic_test_model_evaluations'
            ),
            'component_train_raw_shots': accounting.get('component_train_raw_shots'),
            'component_gradient_raw_shots': accounting.get('component_gradient_raw_shots'),
            'component_training_monitor_raw_shots': accounting.get('component_training_monitor_raw_shots'),
            'component_edge_raw_shots': accounting.get('component_edge_raw_shots'),
            'component_validation_monitor_raw_shots': accounting.get('component_validation_monitor_raw_shots'),
            'component_validation_confirm_raw_shots': accounting.get('component_validation_confirm_raw_shots'),
            'component_train_margin_confirm_model_evaluations': accounting.get(
                'component_train_margin_confirm_model_evaluations'
            ),
            'component_train_margin_confirm_raw_shots': accounting.get(
                'component_train_margin_confirm_raw_shots'
            ),
            'component_diagnostic_test_raw_shots': accounting.get('component_diagnostic_test_raw_shots'),
            'mc_status_mode': accounting.get('mc_status_mode'),
            'mc_refresh_granularity': accounting.get('mc_refresh_granularity'),
            'mc_trajectories_per_expectation': accounting.get('mc_trajectories_per_expectation'),
            'physical_noise_coupling': accounting.get('physical_noise_coupling'),
            'parameter_shift_noise_coupling': accounting.get('parameter_shift_noise_coupling'),
            'double_stochastic_simulation': accounting.get('double_stochastic_simulation'),
            'shot_accounting': accounting,
            'validation_stop_shot_accounting': accounting,
            'final_test_shot_accounting': final_test_accounting,
            'reported_total_shot_accounting': reported_total_accounting,
            'shots_at_validation_stop': accounting['total_raw_shots'],
            'final_test_shots': final_test_accounting['total_raw_shots'],
            'shots_including_final_test': reported_total_accounting['total_raw_shots'],
            'validation_stop_raw_shot_budget': accounting['total_raw_shots'],
            'final_test_raw_shot_budget': final_test_accounting['total_raw_shots'],
            'reported_total_raw_shot_budget': reported_total_accounting['total_raw_shots'],
            'resource_to_confirmed_train_margin_model_evaluations': accounting[
                'total_model_evaluations'
            ],
            'resource_to_confirmed_train_margin_raw_shots': accounting[
                'total_raw_shots'
            ],
        }

    def save(self, path: str, checkpoint_metadata: Optional[Mapping[str, Any]] = None) -> None:
        metadata = dict(checkpoint_metadata or self.checkpoint_metadata or {})
        payload = {
            'algorithm': 'real_adaboost',
            'n_rounds': self.n_rounds,
            'margin_threshold': self.margin_threshold,
            'clip_edge': self.clip_edge,
            'surrogate_sigma': self.surrogate_sigma,
            'use_in_graph_prediction': self.use_in_graph_prediction,
            'test_evaluation_policy': str(self.test_evaluation_policy),
            'validation_confirmation_policy': str(self.validation_confirmation_policy),
            'train_margin_confirmation_policy': str(self.train_margin_confirmation_policy),
            'ensemble_patience_rounds': self.ensemble_patience_rounds,
            'ensemble_patience_min_delta': float(self.ensemble_patience_min_delta),
            'verbose': bool(self.verbose),
            'log_every_rounds': int(self.log_every_rounds),
            'checkpoint_format': 'boosting_qml.real_adaboost.resume.v2',
            'checkpoint_metadata': metadata,
            'checkpoint_every_rounds': int(self.checkpoint_every_rounds),
            'boost_coefs': self.boost_coefs,
            'weak_learner_signs': self.weak_learner_signs,
            'weak_learners': [_serialize_weak_learner_state(learner) for learner in self.weak_learners],
            'round_results': [
                _serialize_round_result(result) for result in self.round_results
            ],
            'latest_resume_state': self._latest_resume_state(),
            'stopping_summary': None if self.stopping_summary is None else asdict(self.stopping_summary),
        }
        path_obj = Path(path)
        _write_json_atomic(path_obj, payload)

    @classmethod
    def load(cls, path: str) -> 'RealAdaBoost':
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if state.get("algorithm") != "real_adaboost":
            raise ValueError("Checkpoint is not a RealAdaBoost checkpoint.")
        if state.get("checkpoint_format") != "boosting_qml.real_adaboost.resume.v2":
            raise ValueError(
                "Checkpoint is not resumable: expected "
                "boosting_qml.real_adaboost.resume.v2 with complete round history."
            )
        model = cls(
            n_rounds=int(state['n_rounds']),
            margin_threshold=float(state['margin_threshold']),
            clip_edge=float(state['clip_edge']),
            surrogate_sigma=float(state.get('surrogate_sigma', 0.5)),
            use_in_graph_prediction=bool(state.get('use_in_graph_prediction', True)),
            test_evaluation_policy=str(state.get('test_evaluation_policy', 'final_only')),
            validation_confirmation_policy=str(
                state.get('validation_confirmation_policy', 'every_round_independent_stream')
            ),
            train_margin_confirmation_policy=str(
                state.get('train_margin_confirmation_policy', 'none')
            ),
            ensemble_patience_rounds=(
                None
                if state.get('ensemble_patience_rounds') is None
                else int(state['ensemble_patience_rounds'])
            ),
            ensemble_patience_min_delta=float(
                state.get('ensemble_patience_min_delta', 0.0)
            ),
            verbose=bool(state.get('verbose', False)),
            log_every_rounds=int(state.get('log_every_rounds', 1)),
            checkpoint_every_rounds=int(state.get('checkpoint_every_rounds', 0)),
            checkpoint_path=str(path),
            checkpoint_metadata=(
                None
                if state.get('checkpoint_metadata') is None
                else dict(state.get('checkpoint_metadata'))
            ),
        )
        model.boost_coefs = [float(v) for v in state['boost_coefs']]
        model.weak_learner_signs = [float(v) for v in state['weak_learner_signs']]

        weak_learner_states = state.get('weak_learners', [])
        if not isinstance(weak_learner_states, list):
            raise ValueError("Checkpoint field 'weak_learners' must be a list.")

        model.weak_learners = []
        for idx, learner_state in enumerate(weak_learner_states):
            if not isinstance(learner_state, Mapping):
                raise ValueError(f"weak_learners[{idx}] must be a mapping, got {type(learner_state).__name__}.")
            model.weak_learners.append(_deserialize_weak_learner_state(learner_state))

        round_states = state.get("round_results")
        if not isinstance(round_states, list):
            raise ValueError("Checkpoint field 'round_results' must be a list.")
        model.round_results = []
        for idx, round_state in enumerate(round_states):
            if not isinstance(round_state, Mapping):
                raise ValueError(
                    f"round_results[{idx}] must be a mapping, got "
                    f"{type(round_state).__name__}."
                )
            model.round_results.append(_deserialize_round_result(round_state))

        resume_state = state.get("latest_resume_state")
        if resume_state is not None and not isinstance(resume_state, Mapping):
            raise ValueError("Checkpoint field 'latest_resume_state' must be an object.")
        model.resumable_state = None if resume_state is None else dict(resume_state)

        stopping_summary_state = state.get('stopping_summary')
        if isinstance(stopping_summary_state, Mapping):
            model.stopping_summary = BoostingStopSummary(**stopping_summary_state)
        model._restored_checkpoint = True
        return model
