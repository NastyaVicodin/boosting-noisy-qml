"""Finite-shot measurement sampling helpers for Pauli expectation scores."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

Array = jax.Array

MeasurementSamplingDistribution = Literal["binomial", "gaussian"]
MeasurementGradientMode = Literal[
    "straight_through_expectation",
    "sample",
    "parameter_shift_binomial",
]


def validate_measurement_sampling_distribution(value: str) -> str:
    """Validate and canonicalize the finite-shot measurement sampler name."""

    value = str(value)
    if value not in {"binomial", "gaussian"}:
        raise ValueError(
            "measurement_sampling_distribution must be one of "
            "['binomial', 'gaussian'], got "
            f"{value!r}."
        )
    return value


def validate_measurement_gradient_mode(value: str) -> str:
    """Validate the gradient convention used for sampled measurements."""

    value = str(value)
    if value not in {
        "straight_through_expectation",
        "sample",
        "parameter_shift_binomial",
    }:
        raise ValueError(
            "measurement_gradient_mode must be one of "
            "['straight_through_expectation', 'sample', 'parameter_shift_binomial'], got "
            f"{value!r}."
        )
    return value


def validate_integer_shots(n_shots: int | None) -> int:
    """Return ``n_shots`` as a positive integer for sampled experiments."""

    if n_shots is None:
        raise ValueError("n_shots must be provided for sampled finite-shot measurement.")
    n_shots_int = int(n_shots)
    if n_shots_int <= 0:
        raise ValueError(f"n_shots must be positive, got {n_shots}.")
    if float(n_shots) != float(n_shots_int):
        raise ValueError(f"n_shots must be an integer for sampled measurement, got {n_shots}.")
    return n_shots_int


def _apply_measurement_gradient_mode(
    sampled_score: Array,
    expectation: Array,
    *,
    gradient_mode: str,
) -> Array:
    gradient_mode = validate_measurement_gradient_mode(gradient_mode)
    sampled_score = jnp.asarray(sampled_score, dtype=jnp.asarray(expectation).dtype)
    expectation = jnp.asarray(expectation)
    if gradient_mode == "straight_through_expectation":
        return expectation + jax.lax.stop_gradient(sampled_score - expectation)
    return jax.lax.stop_gradient(sampled_score)


def sample_pauli_expectation_score(
    expectation: Array,
    key: Array,
    *,
    n_shots: int,
    distribution: str = "binomial",
    gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Sample a finite-shot estimate of a Pauli-valued expectation.

    ``binomial`` implements the exact distribution
    ``N+ ~ Binomial(N_sh, (1 + expectation) / 2)`` and returns
    ``2 N+ / N_sh - 1``.  The straight-through mode keeps this exact sampled
    value in the forward pass while preserving gradients through the underlying
    expectation for the existing autodiff optimizer.
    """

    n_shots_int = validate_integer_shots(n_shots)
    distribution = validate_measurement_sampling_distribution(distribution)
    expectation = jnp.clip(jnp.asarray(expectation), -1.0, 1.0)
    dtype = expectation.dtype

    if distribution == "gaussian":
        var = (1.0 - expectation**2) / float(n_shots_int)
        sampled = expectation + jnp.sqrt(jnp.abs(var)) * jax.random.normal(key, dtype=dtype)
        return jnp.clip(sampled, -1.0, 1.0)

    p_positive = jnp.clip((1.0 + expectation) / 2.0, 0.0, 1.0)
    shot_shape = tuple(expectation.shape) + (int(n_shots_int),)
    outcomes = jax.random.bernoulli(
        key,
        p=jnp.expand_dims(p_positive, axis=-1),
        shape=shot_shape,
    )
    n_positive = jnp.sum(outcomes, axis=-1)
    sampled = 2.0 * n_positive.astype(dtype) / float(n_shots_int) - 1.0
    return _apply_measurement_gradient_mode(
        sampled,
        expectation,
        gradient_mode=gradient_mode,
    )


def sample_single_pauli_outcomes(
    expectations: Array,
    key: Array,
    *,
    gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Sample one Pauli outcome for each supplied expectation value."""

    expectations = jnp.clip(jnp.asarray(expectations), -1.0, 1.0)
    p_positive = jnp.clip((1.0 + expectations) / 2.0, 0.0, 1.0)
    uniforms = jax.random.uniform(
        key,
        shape=expectations.shape,
        minval=0.0,
        maxval=1.0,
        dtype=expectations.dtype,
    )
    sampled = jnp.where(uniforms < p_positive, 1.0, -1.0).astype(expectations.dtype)
    return _apply_measurement_gradient_mode(
        sampled,
        expectations,
        gradient_mode=gradient_mode,
    )


__all__ = [
    "MeasurementGradientMode",
    "MeasurementSamplingDistribution",
    "sample_pauli_expectation_score",
    "sample_single_pauli_outcomes",
    "validate_integer_shots",
    "validate_measurement_gradient_mode",
    "validate_measurement_sampling_distribution",
]
