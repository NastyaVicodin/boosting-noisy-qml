"""Trajectory-level stochastic PEC using an exact QPD marginal sampler.

The explicit gate sampler in :mod:`boosting_qml.pec.finite_sample` returns
``X = gamma * sign(zeta_nu) * Z``.  For a Pauli-Z measurement X is always
``-gamma`` or ``+gamma`` and its mean is the ideal expectation ``mu``.  Thus
the number of positive signed outcomes in N independent QPD executions is
exactly ``Binomial(N, (1 + mu/gamma)/2)``.  Sampling that count is
distributionally identical to materialising N gate-level QPD paths, avoids
shot loops, and is not a Gaussian approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from boosting_qml.pec.finite_sample import (
    inverse_depolarizing_l1_norm_1q,
    inverse_depolarizing_l1_norm_2q,
)
from boosting_qml.rng import stable_int_seed
from boosting_qml.losses.surrogates import PsiG
from boosting_qml.training.finite_shot_parameter_shift import (
    _build_shifted_params_chunk,
    _density_expectation_shift_tensor,
    _loss_grads_from_parameter_shift_tensors,
    legacy_shifted_gate_occurrence_count,
    normalize_parameter_shift_chunk_size,
)


SAMPLED_PEC_IMPLEMENTATION = "qpd_signed_pauli.exact_marginal_binomial.v1"

# Exact circuit expectations are currently produced in float32.  Permit only
# a few unit-scale ULPs of final-rounding excursion before projecting an exact
# Pauli expectation back to its physical interval.  This is intentionally far
# below an excursion that could be scientifically material.
IDEAL_EXPECTATION_PHYSICAL_BOUND_ATOL = 8.0 * float(np.finfo(np.float32).eps)


def ring_entangler_pair_count(n_qubits: int) -> int:
    n = int(n_qubits)
    if n < 2:
        raise ValueError("n_qubits must be at least two.")
    return n if n > 2 else 1


def legacy_reuploading_gate_counts(n_qubits: int, n_layers: int) -> tuple[int, int]:
    n, depth = int(n_qubits), int(n_layers)
    if n <= 0 or depth <= 0:
        raise ValueError("n_qubits and n_layers must be positive.")
    return (4 * depth + 3) * n, depth * ring_entangler_pair_count(n)


def circuit_pec_gamma(
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool = True,
) -> float:
    """Circuit QPD L1 norm used by the explicit PEC implementation."""

    g1, g2 = legacy_reuploading_gate_counts(n_qubits, n_layers)
    gamma = inverse_depolarizing_l1_norm_1q(noise_strength_1q) ** g1
    if apply_two_qubit_noise:
        gamma *= inverse_depolarizing_l1_norm_2q(noise_strength_2q) ** g2
    return float(gamma)


def variance_matched_raw_samples(n_effective: int, gamma_squared: float) -> int:
    n_eff, gamma_sq = int(n_effective), float(gamma_squared)
    if n_eff <= 0:
        raise ValueError("n_effective must be positive.")
    if not math.isfinite(gamma_sq) or gamma_sq < 1.0:
        raise ValueError("gamma_squared must be finite and at least one.")
    return int(math.ceil(n_eff * gamma_sq))


@dataclass(frozen=True)
class SampledPECProtocol:
    n_qubits: int
    n_layers: int
    noise_strength_1q: float
    noise_strength_2q: float
    n_effective: int = 1000
    apply_two_qubit_noise: bool = True

    @property
    def gate_counts(self) -> tuple[int, int]:
        return legacy_reuploading_gate_counts(self.n_qubits, self.n_layers)

    @property
    def gamma(self) -> float:
        return circuit_pec_gamma(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            noise_strength_1q=self.noise_strength_1q,
            noise_strength_2q=self.noise_strength_2q,
            apply_two_qubit_noise=self.apply_two_qubit_noise,
        )

    @property
    def gamma_squared(self) -> float:
        return float(self.gamma * self.gamma)

    @property
    def raw_samples_per_expectation(self) -> int:
        return variance_matched_raw_samples(self.n_effective, self.gamma_squared)

    def to_dict(self) -> dict[str, Any]:
        g1, g2 = self.gate_counts
        return {
            "implementation_identity": SAMPLED_PEC_IMPLEMENTATION,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "noise_strength_1q": self.noise_strength_1q,
            "noise_strength_2q": self.noise_strength_2q,
            "apply_two_qubit_noise": self.apply_two_qubit_noise,
            "one_qubit_gate_count_G1": g1,
            "two_qubit_gate_count_G2": g2,
            "one_qubit_qpd_norm": inverse_depolarizing_l1_norm_1q(self.noise_strength_1q),
            "two_qubit_qpd_norm": inverse_depolarizing_l1_norm_2q(self.noise_strength_2q),
            "circuit_gamma": self.gamma,
            "circuit_gamma_squared": self.gamma_squared,
            "n_effective": self.n_effective,
            "n_raw_pec": self.raw_samples_per_expectation,
            "raw_sample_rule": "ceil(n_effective * circuit_gamma_squared)",
        }


def sample_signed_pec_mean(
    ideal_expectation: np.ndarray | float,
    *,
    protocol: SampledPECProtocol,
    seeds: np.ndarray | int,
) -> np.ndarray:
    """Sample exact averages of the signed QPD estimator without shot loops."""

    mu = _sanitize_ideal_expectation(ideal_expectation)
    seed_array = np.broadcast_to(np.asarray(seeds, dtype=np.uint64), mu.shape)
    gamma, n_raw = protocol.gamma, protocol.raw_samples_per_expectation
    probability = np.clip((1.0 + mu / gamma) / 2.0, 0.0, 1.0)
    output = np.empty(mu.shape, dtype=np.float64)
    for index in np.ndindex(mu.shape):
        count = np.random.default_rng(int(seed_array[index])).binomial(
            n=n_raw, p=float(probability[index])
        )
        output[index] = gamma * (2.0 * count / n_raw - 1.0)
    return output


def _sanitize_ideal_expectation(
    ideal_expectation: np.ndarray | float,
) -> np.ndarray:
    """Validate and project only machine-scale Pauli-bound excursions.

    The exact simulator path feeding this estimator evaluates in float32, for
    which a physically bounded expectation can finish a few ULPs outside
    ``[-1, 1]``.  Non-finite values and excursions larger than the explicit
    float32-scale tolerance remain hard failures.
    """

    mu = np.asarray(ideal_expectation, dtype=np.float64)
    if np.any(~np.isfinite(mu)):
        raise ValueError("ideal_expectation must be finite.")
    excess = np.maximum(np.abs(mu) - 1.0, 0.0)
    if np.any(excess > IDEAL_EXPECTATION_PHYSICAL_BOUND_ATOL):
        raise ValueError(
            "ideal_expectation is materially outside [-1, 1] "
            f"(tolerance={IDEAL_EXPECTATION_PHYSICAL_BOUND_ATOL:.17g})."
        )
    return np.clip(mu, -1.0, 1.0)


def estimator_variance(ideal_expectation: float, protocol: SampledPECProtocol) -> float:
    mu = float(ideal_expectation)
    if abs(mu) > 1.0:
        raise ValueError("ideal_expectation must lie in [-1, 1].")
    return float((protocol.gamma_squared - mu * mu) / protocol.raw_samples_per_expectation)


def _semantic_seeds(
    shape: tuple[int, ...], *, base_seed: int, stream: str, offset: int = 0
) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint64)
    for flat_index in range(int(np.prod(shape))):
        result.flat[flat_index] = stable_int_seed(
            "sampled_pec_exact_marginal", base_seed, stream, offset + flat_index
        )
    return result


def sampled_pec_predict_batch(
    params: jax.Array,
    inputs: jax.Array,
    *,
    protocol: SampledPECProtocol,
    base_seed: int,
    stream: str,
    observable: str = "global_z",
) -> tuple[jax.Array, dict[str, int]]:
    params, inputs = jnp.asarray(params, dtype=jnp.float32), jnp.asarray(inputs, dtype=jnp.float32)
    ideal = _density_expectation_shift_tensor(
        jnp.broadcast_to(params, (1, int(inputs.shape[0])) + tuple(params.shape)),
        inputs,
        n_qubits=protocol.n_qubits,
        n_layers=protocol.n_layers,
        noise_strength_1q=0.0,
        noise_strength_2q=0.0,
        apply_two_qubit_noise=protocol.apply_two_qubit_noise,
        observable=observable,
    )[0]
    ideal_np = np.asarray(jax.device_get(ideal), dtype=np.float64)
    sampled = sample_signed_pec_mean(
        ideal_np,
        protocol=protocol,
        seeds=_semantic_seeds(ideal_np.shape, base_seed=base_seed, stream=stream),
    )
    n_expectations = int(ideal_np.size)
    return jnp.asarray(sampled, dtype=params.dtype), {
        "expectation_estimations": n_expectations,
        "raw_physical_shots": n_expectations * protocol.raw_samples_per_expectation,
    }


def _ideal_scores(
    params: jax.Array,
    inputs: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    apply_two_qubit_noise: bool,
    observable: str,
) -> jax.Array:
    """Exact ideal expectations underlying the unbiased PEC estimator."""

    return _density_expectation_shift_tensor(
        jnp.broadcast_to(params, (1, int(inputs.shape[0])) + tuple(params.shape)),
        inputs,
        n_qubits=n_qubits,
        n_layers=n_layers,
        noise_strength_1q=0.0,
        noise_strength_2q=0.0,
        apply_two_qubit_noise=apply_two_qubit_noise,
        observable=observable,
    )[0]


def sampled_pec_straight_through_loss_gradient(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    *,
    protocol: SampledPECProtocol,
    base_seed: int,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    observable: str = "global_z",
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, int | str]]:
    """Strict-v5 straight-through gradient for stochastic PEC forward scores.

    The forward loss is evaluated at a finite-sample PEC score.  The backward
    pass differentiates the corresponding exact ideal circuit expectation:
    ``d loss/d score`` at the sampled score times ``d h_ideal/d params``.
    No shifted parameter circuits are evaluated or charged.  This is a
    simulation-side surrogate and is not a hardware-compatible PEC gradient.
    """

    params = jnp.asarray(params, dtype=jnp.float32)
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.float32)
    ideal_function = lambda current_params: _ideal_scores(
        current_params,
        inputs,
        n_qubits=protocol.n_qubits,
        n_layers=protocol.n_layers,
        apply_two_qubit_noise=protocol.apply_two_qubit_noise,
        observable=observable,
    )
    ideal_at_current, ideal_pullback = jax.vjp(ideal_function, params)
    ideal_np = np.asarray(jax.device_get(ideal_at_current), dtype=np.float64)
    sampled_np = sample_signed_pec_mean(
        ideal_np,
        protocol=protocol,
        seeds=_semantic_seeds(
            ideal_np.shape,
            base_seed=base_seed,
            stream="strict_v5_straight_through_forward",
        ),
    )
    sampled = jnp.asarray(sampled_np, dtype=params.dtype)

    def objective(scores: jax.Array) -> jax.Array:
        per_example = PsiG(labels * scores, sigma)
        if weights is None:
            return jnp.mean(per_example)
        else:
            normalized = jnp.asarray(weights, dtype=per_example.dtype)
            normalized = normalized / jnp.sum(normalized)
            return jnp.sum(normalized * per_example)

    loss, score_cotangent = jax.value_and_grad(objective)(sampled)
    gradients = ideal_pullback(score_cotangent)[0]
    n_expectations = int(inputs.shape[0])
    return gradients, loss, sampled, {
        "unshifted_expectation_estimations": n_expectations,
        "shifted_expectation_estimations": 0,
        "expectation_estimations": n_expectations,
        "raw_physical_shots": n_expectations * protocol.raw_samples_per_expectation,
        "gradient_semantics": "strict_v5_simulation_side_straight_through_ideal_expectation",
    }


def sampled_pec_parameter_shift_loss_gradient(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    *,
    protocol: SampledPECProtocol,
    base_seed: int,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    observable: str = "global_z",
    parameter_shift_chunk_size: int | str | None = 16,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, int]]:
    """Plug-in parameter-shift gradient with independent QPD streams."""

    params = jnp.asarray(params, dtype=jnp.float32)
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.float32)
    shifted_count = legacy_shifted_gate_occurrence_count(
        n_qubits=protocol.n_qubits, n_layers=protocol.n_layers
    )
    chunk_size = normalize_parameter_shift_chunk_size(parameter_shift_chunk_size, shifted_count)
    n_examples = int(inputs.shape[0])
    ideal0 = _density_expectation_shift_tensor(
        jnp.broadcast_to(params, (1, n_examples) + tuple(params.shape)),
        inputs,
        n_qubits=protocol.n_qubits,
        n_layers=protocol.n_layers,
        noise_strength_1q=0.0,
        noise_strength_2q=0.0,
        apply_two_qubit_noise=protocol.apply_two_qubit_noise,
        observable=observable,
    )[0]
    ideal0_np = np.asarray(jax.device_get(ideal0), dtype=np.float64)
    sampled0 = sample_signed_pec_mean(
        ideal0_np,
        protocol=protocol,
        seeds=_semantic_seeds(ideal0_np.shape, base_seed=base_seed, stream="gradient_unshifted"),
    )
    plus_parts, minus_parts = [], []
    for start in range(0, shifted_count, chunk_size):
        stop = min(start + chunk_size, shifted_count)
        shifted_params = _build_shifted_params_chunk(
            params,
            inputs,
            jnp.arange(start, stop, dtype=jnp.int32),
            n_qubits=protocol.n_qubits,
            n_layers=protocol.n_layers,
        )
        ideal_chunk = _density_expectation_shift_tensor(
            shifted_params,
            inputs,
            n_qubits=protocol.n_qubits,
            n_layers=protocol.n_layers,
            noise_strength_1q=0.0,
            noise_strength_2q=0.0,
            apply_two_qubit_noise=protocol.apply_two_qubit_noise,
            observable=observable,
        )
        chunk_np = np.asarray(jax.device_get(ideal_chunk), dtype=np.float64)
        n_chunk = stop - start
        for name, values, target in (
            ("gradient_plus", chunk_np[:n_chunk], plus_parts),
            ("gradient_minus", chunk_np[n_chunk:], minus_parts),
        ):
            target.append(
                jnp.asarray(
                    sample_signed_pec_mean(
                        values,
                        protocol=protocol,
                        seeds=_semantic_seeds(
                            values.shape,
                            base_seed=base_seed,
                            stream=name,
                            offset=start * n_examples,
                        ),
                    ),
                    dtype=params.dtype,
                )
            )
    tensors = {
        "unshifted_scores": jnp.asarray(sampled0, dtype=params.dtype),
        "plus_scores": jnp.concatenate(plus_parts, axis=0),
        "minus_scores": jnp.concatenate(minus_parts, axis=0),
    }
    grads, loss, preds = _loss_grads_from_parameter_shift_tensors(
        tensors,
        inputs,
        labels,
        weights,
        params_shape=tuple(params.shape),
        n_qubits=protocol.n_qubits,
        n_layers=protocol.n_layers,
        sigma=sigma,
    )
    n_expectations = n_examples * (1 + 2 * shifted_count)
    return grads, loss, preds, {
        "unshifted_expectation_estimations": n_examples,
        "shifted_expectation_estimations": n_examples * 2 * shifted_count,
        "expectation_estimations": n_expectations,
        "raw_physical_shots": n_expectations * protocol.raw_samples_per_expectation,
    }


__all__ = [
    "SAMPLED_PEC_IMPLEMENTATION",
    "SampledPECProtocol",
    "circuit_pec_gamma",
    "estimator_variance",
    "legacy_reuploading_gate_counts",
    "sample_signed_pec_mean",
    "sampled_pec_parameter_shift_loss_gradient",
    "sampled_pec_predict_batch",
    "sampled_pec_straight_through_loss_gradient",
    "variance_matched_raw_samples",
]
