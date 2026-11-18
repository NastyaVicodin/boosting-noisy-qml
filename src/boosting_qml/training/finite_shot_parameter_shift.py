"""Finite-shot parameter-shift gradients for density-matrix PQC training."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from boosting_qml.losses.surrogates import PsiG, standard_normal_cdf
from boosting_qml.pqc.ansatz import DTYPE
from boosting_qml.pqc.density_matrix import exact_noisy_reuploading_density_score
from boosting_qml.pqc.measurement import (
    validate_integer_shots,
    validate_measurement_gradient_mode,
    validate_measurement_sampling_distribution,
)
from boosting_qml.rng import stable_int_seed

_PARAM_SHIFT_COEFF_EPS = 1e-7


def legacy_shifted_gate_occurrence_count(*, n_qubits: int, n_layers: int) -> int:
    return int((4 * int(n_layers) + 3) * int(n_qubits))


def normalize_parameter_shift_chunk_size(chunk_size: int | str | None, shifted_count: int) -> int:
    if chunk_size is None:
        return int(shifted_count)
    if isinstance(chunk_size, str):
        value = chunk_size.strip().lower()
        if value in {"", "0", "all", "none"}:
            return int(shifted_count)
        parsed = int(value)
    else:
        parsed = int(chunk_size)
    if parsed <= 0 or parsed >= int(shifted_count):
        return int(shifted_count)
    return int(parsed)


@lru_cache(maxsize=128)
def _cached_density_expectation_batch(
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
):
    @jax.jit
    def _predict(params: jax.Array, inputs: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda x_i: exact_noisy_reuploading_density_score(
                x_i,
                n_qubits=int(n_qubits),
                params=params,
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
        )(inputs)

    return _predict


@lru_cache(maxsize=128)
def _cached_density_expectation_batch_per_example(
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
):
    @jax.jit
    def _predict(params_per_example: jax.Array, inputs: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda params_i, x_i: exact_noisy_reuploading_density_score(
                x_i,
                n_qubits=int(n_qubits),
                params=params_i,
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
        )(params_per_example, inputs)

    return _predict


@lru_cache(maxsize=128)
def _cached_density_expectation_shift_tensor(
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
):
    @jax.jit
    def _predict(params_tensor: jax.Array, inputs: jax.Array) -> jax.Array:
        # params_tensor has shape [n_shift_rows, batch, param_rows, n_qubits].
        # ``lax.map`` over shifted rows avoids a giant HLO graph over all
        # parameter occurrences, while the inner ``vmap`` keeps the batch of
        # examples vectorized for each shifted row.
        def _row_body(params_for_row: jax.Array) -> jax.Array:
            return jax.vmap(
                lambda params_i, x_i: exact_noisy_reuploading_density_score(
                    x_i,
                    n_qubits=int(n_qubits),
                    params=params_i,
                    n_layers=int(n_layers),
                    noise_strength_1q=float(noise_strength_1q),
                    noise_strength_2q=float(noise_strength_2q),
                    apply_two_qubit_noise=bool(apply_two_qubit_noise),
                    observable=str(observable),
                )
            )(params_for_row, inputs)

        return jax.lax.map(_row_body, params_tensor)

    return _predict


def _key_parts(key: jax.Array) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(jax.device_get(key), dtype=np.uint32).reshape(-1))


def _validate_common_inputs(
    params: jax.Array,
    inputs: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    ansatz_kind: str,
    noise_backend: str,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
) -> int:
    if str(ansatz_kind) != "legacy_reuploading":
        raise ValueError("parameter_shift_binomial currently supports only legacy_reuploading.")
    if str(noise_backend) != "density_matrix":
        raise ValueError("parameter_shift_binomial currently supports only density_matrix.")
    if validate_measurement_sampling_distribution(measurement_sampling_distribution) != "binomial":
        raise ValueError("parameter_shift_binomial requires binomial measurement sampling.")
    if validate_measurement_gradient_mode(measurement_gradient_mode) != "parameter_shift_binomial":
        raise ValueError("Expected measurement_gradient_mode='parameter_shift_binomial'.")
    n_shots_int = validate_integer_shots(int(n_shots))
    expected_shape = (4 * int(n_layers) + 3, int(n_qubits))
    if tuple(params.shape) != expected_shape:
        raise ValueError(f"Expected params shape {expected_shape}, got {tuple(params.shape)}.")
    if int(inputs.shape[1]) != int(n_qubits):
        raise ValueError(f"Expected input dimension {n_qubits}, got {inputs.shape[1]}.")
    if int(inputs.shape[0]) <= 0:
        raise ValueError("Parameter-shift training batch must contain at least one example.")
    return n_shots_int


def _density_expectations(
    params: jax.Array,
    inputs: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
) -> jax.Array:
    predict = _cached_density_expectation_batch(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=str(observable),
    )
    return predict(params, inputs)


def _density_expectations_per_example(
    params_per_example: jax.Array,
    inputs: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
) -> jax.Array:
    predict = _cached_density_expectation_batch_per_example(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=str(observable),
    )
    return predict(params_per_example, inputs)


def _density_expectation_shift_tensor(
    params_tensor: jax.Array,
    inputs: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
) -> jax.Array:
    predict = _cached_density_expectation_shift_tensor(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=str(observable),
    )
    return predict(params_tensor, inputs)


def _semantic_seed_matrix(
    *,
    base_parts: tuple[Any, ...],
    row_kinds: list[tuple[int | None, str]],
    n_examples: int,
) -> np.ndarray:
    seeds = np.empty((len(row_kinds), int(n_examples)), dtype=np.uint32)
    for row_idx, (param_index, shift_sign) in enumerate(row_kinds):
        for example_index in range(int(n_examples)):
            if param_index is None:
                seed = stable_int_seed(
                    "finite_shot_parameter_shift",
                    *base_parts,
                    "unshifted",
                    "example",
                    example_index,
                )
            else:
                seed = stable_int_seed(
                    "finite_shot_parameter_shift",
                    *base_parts,
                    "parameter",
                    int(param_index),
                    str(shift_sign),
                    "example",
                    example_index,
                )
            seeds[row_idx, example_index] = np.uint32(seed)
    return seeds


def _sample_semantic_binomial_score_tensor(
    expectations: jax.Array,
    *,
    n_shots: int,
    dtype: Any,
    base_parts: tuple[Any, ...],
    row_kinds: list[tuple[int | None, str]],
) -> jax.Array:
    n_shots_int = validate_integer_shots(int(n_shots))
    expectations = jnp.clip(jnp.asarray(expectations, dtype=dtype), -1.0, 1.0)
    seed_matrix = _semantic_seed_matrix(
        base_parts=base_parts,
        row_kinds=row_kinds,
        n_examples=int(expectations.shape[1]),
    )
    seeds = jnp.asarray(seed_matrix, dtype=jnp.uint32)
    keys = jax.vmap(jax.vmap(jax.random.PRNGKey))(seeds)
    flat_keys = keys.reshape((-1, 2))
    flat_p = jnp.clip(((1.0 + expectations) / 2.0).reshape(-1), 0.0, 1.0)
    flat_counts = jax.vmap(
        lambda key_i, p_i: jax.random.binomial(key_i, n=n_shots_int, p=p_i)
    )(flat_keys, flat_p)
    sampled = 2.0 * flat_counts.astype(dtype) / float(n_shots_int) - 1.0
    return sampled.reshape(expectations.shape)


def _build_shifted_params_chunk(
    params: jax.Array,
    inputs: jax.Array,
    flat_indices: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
) -> jax.Array:
    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    flat_indices = jnp.asarray(flat_indices, dtype=jnp.int32)
    n_chunk = int(flat_indices.shape[0])
    n_examples = int(inputs.shape[0])

    rows = flat_indices // int(n_qubits)
    cols = flat_indices - rows * int(n_qubits)
    is_encoding = jnp.logical_and(rows < int(4 * n_layers), (rows % 4) == 0)
    coeff = jnp.take(inputs, cols, axis=1).T
    active = jnp.logical_or(
        jnp.logical_not(is_encoding)[:, None],
        jnp.abs(coeff) > jnp.asarray(_PARAM_SHIFT_COEFF_EPS, dtype=inputs.dtype),
    )
    safe_coeff = jnp.where(active, coeff, 1.0)
    delta_abs = jnp.where(
        is_encoding[:, None],
        jnp.pi / (2.0 * safe_coeff),
        jnp.pi / 2.0,
    ).astype(params.dtype)
    delta_abs = jnp.where(active, delta_abs, 0.0)

    params_tensor = jnp.broadcast_to(
        params,
        (2, n_chunk, n_examples) + tuple(params.shape),
    )
    sign_values = jnp.asarray([1.0, -1.0], dtype=params.dtype)[:, None, None]
    delta = sign_values * delta_abs[None, :, :]

    sign_idx = jnp.arange(2, dtype=jnp.int32)[:, None, None]
    chunk_idx = jnp.arange(n_chunk, dtype=jnp.int32)[None, :, None]
    example_idx = jnp.arange(n_examples, dtype=jnp.int32)[None, None, :]
    row_idx = rows[None, :, None]
    col_idx = cols[None, :, None]
    params_tensor = params_tensor.at[
        sign_idx,
        chunk_idx,
        example_idx,
        row_idx,
        col_idx,
    ].add(delta)
    return params_tensor.reshape((2 * n_chunk, n_examples) + tuple(params.shape))


def _parameter_shift_coefficients(
    inputs: jax.Array,
    flat_indices: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
) -> jax.Array:
    rows = flat_indices // int(n_qubits)
    cols = flat_indices - rows * int(n_qubits)
    is_encoding = jnp.logical_and(rows < int(4 * n_layers), (rows % 4) == 0)
    coeff = jnp.take(inputs, cols, axis=1).T
    active = jnp.abs(coeff) > jnp.asarray(_PARAM_SHIFT_COEFF_EPS, dtype=inputs.dtype)
    return jnp.where(is_encoding[:, None], jnp.where(active, coeff, 0.0), 1.0)


def vectorized_parameter_shift_tensors_density(
    params: jax.Array,
    inputs: jax.Array,
    key: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    ansatz_kind: str = "legacy_reuploading",
    noise_backend: str = "density_matrix",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "parameter_shift_binomial",
    observable: str = "global_z",
    parameter_shift_chunk_size: int | str | None = None,
) -> dict[str, jax.Array]:
    """Return exact and sampled unshifted/plus/minus tensors.

    Production code may loop over a small number of coarse chunks, but each
    chunk evaluates all examples and both signs for its parameter subset as one
    compiled shifted tensor.
    """

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    n_shots_int = _validate_common_inputs(
        params,
        inputs,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots=int(n_shots),
        ansatz_kind=ansatz_kind,
        noise_backend=noise_backend,
        measurement_sampling_distribution=measurement_sampling_distribution,
        measurement_gradient_mode=measurement_gradient_mode,
    )
    dtype = params.dtype
    base_parts = _key_parts(key)
    n_examples = int(inputs.shape[0])
    shifted_count = legacy_shifted_gate_occurrence_count(n_qubits=int(n_qubits), n_layers=int(n_layers))
    chunk_size = normalize_parameter_shift_chunk_size(parameter_shift_chunk_size, shifted_count)

    unshifted_params = jnp.broadcast_to(params, (1, n_examples) + tuple(params.shape))
    unshifted_expectations = _density_expectation_shift_tensor(
        unshifted_params,
        inputs,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=str(observable),
    )[0]
    unshifted_scores = _sample_semantic_binomial_score_tensor(
        unshifted_expectations[None, :],
        n_shots=n_shots_int,
        dtype=dtype,
        base_parts=base_parts,
        row_kinds=[(None, "unshifted")],
    )[0]

    plus_expectations_parts: list[jax.Array] = []
    minus_expectations_parts: list[jax.Array] = []
    plus_scores_parts: list[jax.Array] = []
    minus_scores_parts: list[jax.Array] = []
    for start in range(0, shifted_count, chunk_size):
        stop = min(start + chunk_size, shifted_count)
        flat_indices = jnp.arange(start, stop, dtype=jnp.int32)
        shifted_params = _build_shifted_params_chunk(
            params,
            inputs,
            flat_indices,
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
        )
        expectations_chunk = _density_expectation_shift_tensor(
            shifted_params,
            inputs,
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
            noise_strength_1q=float(noise_strength_1q),
            noise_strength_2q=float(noise_strength_2q),
            apply_two_qubit_noise=bool(apply_two_qubit_noise),
            observable=str(observable),
        )
        chunk_indices = list(range(start, stop))
        row_kinds = (
            [(idx, "plus") for idx in chunk_indices]
            + [(idx, "minus") for idx in chunk_indices]
        )
        sampled_chunk = _sample_semantic_binomial_score_tensor(
            expectations_chunk,
            n_shots=n_shots_int,
            dtype=dtype,
            base_parts=base_parts,
            row_kinds=row_kinds,
        )
        n_chunk = stop - start
        plus_expectations_parts.append(expectations_chunk[:n_chunk])
        minus_expectations_parts.append(expectations_chunk[n_chunk:])
        plus_scores_parts.append(sampled_chunk[:n_chunk])
        minus_scores_parts.append(sampled_chunk[n_chunk:])

    return {
        "unshifted_expectations": unshifted_expectations,
        "plus_expectations": jnp.concatenate(plus_expectations_parts, axis=0),
        "minus_expectations": jnp.concatenate(minus_expectations_parts, axis=0),
        "unshifted_scores": unshifted_scores,
        "plus_scores": jnp.concatenate(plus_scores_parts, axis=0),
        "minus_scores": jnp.concatenate(minus_scores_parts, axis=0),
    }


def _reference_host_loop_parameter_shift_tensors_density(
    params: jax.Array,
    inputs: jax.Array,
    key: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    ansatz_kind: str = "legacy_reuploading",
    noise_backend: str = "density_matrix",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "parameter_shift_binomial",
    observable: str = "global_z",
) -> dict[str, jax.Array]:
    """Private reference implementation retained for vectorization tests only."""

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    n_shots_int = _validate_common_inputs(
        params,
        inputs,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots=int(n_shots),
        ansatz_kind=ansatz_kind,
        noise_backend=noise_backend,
        measurement_sampling_distribution=measurement_sampling_distribution,
        measurement_gradient_mode=measurement_gradient_mode,
    )
    dtype = params.dtype
    base_parts = _key_parts(key)
    shifted_count = legacy_shifted_gate_occurrence_count(n_qubits=int(n_qubits), n_layers=int(n_layers))
    flat_params = params.reshape(-1)
    params_np = np.asarray(jax.device_get(params), dtype=np.float32)
    inputs_np = np.asarray(jax.device_get(inputs), dtype=np.float32)
    n_examples = int(inputs.shape[0])

    unshifted_expectations = _density_expectations(
        params,
        inputs,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=str(observable),
    )
    plus_expectations: list[jax.Array] = []
    minus_expectations: list[jax.Array] = []
    for flat_index in range(shifted_count):
        row = flat_index // int(n_qubits)
        col = flat_index - row * int(n_qubits)
        is_encoding_param = row < int(4 * n_layers) and (row % 4) == 0

        if is_encoding_param:
            coeff_np = inputs_np[:, col].astype(np.float32)
            active_np = np.abs(coeff_np) > _PARAM_SHIFT_COEFF_EPS
            safe_coeff_np = np.where(active_np, coeff_np, 1.0).astype(np.float32)
            delta_np = (np.pi / (2.0 * safe_coeff_np)).astype(np.float32)
            delta_np = np.where(active_np, delta_np, 0.0).astype(np.float32)
            plus_np = np.broadcast_to(params_np, (n_examples,) + params_np.shape).copy()
            minus_np = np.broadcast_to(params_np, (n_examples,) + params_np.shape).copy()
            plus_np[:, row, col] += delta_np
            minus_np[:, row, col] -= delta_np
            plus = _density_expectations_per_example(
                jnp.asarray(plus_np, dtype=dtype),
                inputs,
                n_qubits=int(n_qubits),
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
            minus = _density_expectations_per_example(
                jnp.asarray(minus_np, dtype=dtype),
                inputs,
                n_qubits=int(n_qubits),
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
        else:
            shift = jnp.zeros_like(flat_params).at[flat_index].set(jnp.asarray(jnp.pi / 2.0, dtype=dtype))
            plus = _density_expectations(
                (flat_params + shift).reshape(params.shape),
                inputs,
                n_qubits=int(n_qubits),
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
            minus = _density_expectations(
                (flat_params - shift).reshape(params.shape),
                inputs,
                n_qubits=int(n_qubits),
                n_layers=int(n_layers),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                apply_two_qubit_noise=bool(apply_two_qubit_noise),
                observable=str(observable),
            )
        plus_expectations.append(plus)
        minus_expectations.append(minus)

    plus_expectations_arr = jnp.stack(plus_expectations, axis=0)
    minus_expectations_arr = jnp.stack(minus_expectations, axis=0)
    row_kinds = (
        [(None, "unshifted")]
        + [(idx, "plus") for idx in range(shifted_count)]
        + [(idx, "minus") for idx in range(shifted_count)]
    )
    sampled = _sample_semantic_binomial_score_tensor(
        jnp.concatenate(
            [
                unshifted_expectations[None, :],
                plus_expectations_arr,
                minus_expectations_arr,
            ],
            axis=0,
        ),
        n_shots=n_shots_int,
        dtype=dtype,
        base_parts=base_parts,
        row_kinds=row_kinds,
    )
    return {
        "unshifted_expectations": unshifted_expectations,
        "plus_expectations": plus_expectations_arr,
        "minus_expectations": minus_expectations_arr,
        "unshifted_scores": sampled[0],
        "plus_scores": sampled[1 : 1 + shifted_count],
        "minus_scores": sampled[1 + shifted_count :],
    }


def _loss_grads_from_parameter_shift_tensors(
    tensors: dict[str, jax.Array],
    inputs: jax.Array,
    labels: jax.Array,
    weights: jax.Array | None,
    *,
    params_shape: tuple[int, int],
    n_qubits: int,
    n_layers: int,
    sigma: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dtype = jnp.asarray(tensors["unshifted_scores"]).dtype
    n_examples = int(inputs.shape[0])
    shifted_count = legacy_shifted_gate_occurrence_count(n_qubits=int(n_qubits), n_layers=int(n_layers))
    if weights is None:
        weights_norm = jnp.ones((n_examples,), dtype=dtype) / float(n_examples)
    else:
        weights_arr = jnp.clip(jnp.asarray(weights, dtype=dtype), jnp.asarray(np.finfo(np.float64).eps, dtype=dtype))
        weights_norm = weights_arr / jnp.sum(weights_arr)

    preds0 = jnp.asarray(tensors["unshifted_scores"], dtype=dtype)
    labels = jnp.asarray(labels, dtype=dtype)
    inputs = jnp.asarray(inputs, dtype=dtype)
    alpha = labels * preds0
    loss = jnp.sum(weights_norm * PsiG(alpha, sigma=float(sigma)))
    dloss_dscore = weights_norm * standard_normal_cdf((1.0 - alpha) / float(sigma)) * (-labels)

    flat_indices = jnp.arange(shifted_count, dtype=jnp.int32)
    coeffs = _parameter_shift_coefficients(
        inputs,
        flat_indices,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
    )
    dscore = 0.5 * (tensors["plus_scores"] - tensors["minus_scores"])
    dscore = coeffs * dscore
    grads_flat = jnp.sum(dloss_dscore[None, :] * dscore, axis=1)
    grads = grads_flat.reshape(params_shape)
    return grads, loss, preds0


def finite_shot_plugin_loss_grads_and_preds_density(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    ansatz_kind: str = "legacy_reuploading",
    noise_backend: str = "density_matrix",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "parameter_shift_binomial",
    observable: str = "global_z",
    parameter_shift_chunk_size: int | str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return finite-shot plug-in loss, params gradient, and unshifted scores."""

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    labels = jnp.asarray(labels, dtype=DTYPE)
    tensors = vectorized_parameter_shift_tensors_density(
        params,
        inputs,
        key,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots=int(n_shots),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        ansatz_kind=str(ansatz_kind),
        noise_backend=str(noise_backend),
        measurement_sampling_distribution=str(measurement_sampling_distribution),
        measurement_gradient_mode=str(measurement_gradient_mode),
        observable=str(observable),
        parameter_shift_chunk_size=parameter_shift_chunk_size,
    )
    return _loss_grads_from_parameter_shift_tensors(
        tensors,
        inputs,
        labels,
        weights,
        params_shape=tuple(params.shape),
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        sigma=float(sigma),
    )


def _reference_host_loop_loss_grads_and_preds_density(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    ansatz_kind: str = "legacy_reuploading",
    noise_backend: str = "density_matrix",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "parameter_shift_binomial",
    observable: str = "global_z",
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array]:
    """Reference loop used only by tests and audits."""

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    labels = jnp.asarray(labels, dtype=DTYPE)
    tensors = _reference_host_loop_parameter_shift_tensors_density(
        params,
        inputs,
        key,
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots=int(n_shots),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        ansatz_kind=str(ansatz_kind),
        noise_backend=str(noise_backend),
        measurement_sampling_distribution=str(measurement_sampling_distribution),
        measurement_gradient_mode=str(measurement_gradient_mode),
        observable=str(observable),
    )
    grads, loss, preds = _loss_grads_from_parameter_shift_tensors(
        tensors,
        inputs,
        labels,
        weights,
        params_shape=tuple(params.shape),
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        sigma=float(sigma),
    )
    return tensors, grads, loss, preds


__all__ = [
    "_reference_host_loop_loss_grads_and_preds_density",
    "finite_shot_plugin_loss_grads_and_preds_density",
    "legacy_shifted_gate_occurrence_count",
    "normalize_parameter_shift_chunk_size",
    "vectorized_parameter_shift_tensors_density",
]
