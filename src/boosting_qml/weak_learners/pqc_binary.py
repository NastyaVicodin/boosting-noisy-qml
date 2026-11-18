"""PQC weak learner aligned with EM ansatz choices."""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np
import optax

from boosting_qml.pqc.monte_carlo import (
    legacy_reuploading_noisy_gate_counts,
    make_reuploading_mc_statuses,
    noisy_reuploading_sampled_score_streaming_status,
    noisy_reuploading_sampled_score_vmap_status,
    noisy_reuploading_score_streaming_status,
    noisy_reuploading_score_vmap_status,
    sample_noisy_reuploading_score_streaming,
    sample_noisy_reuploading_score_vmap,
)
from boosting_qml.pqc.measurement import (
    sample_pauli_expectation_score,
    validate_integer_shots,
    validate_measurement_gradient_mode,
    validate_measurement_sampling_distribution,
)
from boosting_qml.pec import build_shot_accounting_summary
from boosting_qml.rng import weighted_pqc_fit_seed
from boosting_qml.training.convergence import (
    BEST_LOSS_NO_IMPROVEMENT_DEFINITION,
    BestLossNoImprovementConfig,
    BestLossNoImprovementTracker,
    WindowedPlateauConfig,
    WindowedPlateauTracker,
)
from boosting_qml.training.finite_shot_parameter_shift import (
    finite_shot_plugin_loss_grads_and_preds_density,
)
from boosting_qml.types import Array, BinaryDataset
from boosting_qml.weak_learners.base import BinaryWeakLearner, WeakLearnerFitResult

# Reuse the exact ansatz/initialization implementations from phase-0.5 EM
# so boosting and EM stay in the same model family.
from boosting_qml.pqc.ansatz import (
    RX_RZ_RY_RZ_layer_scan,
    initialize_paper_modular_params,
    initialize_reuploading_params,
    paper_modular_reuploading_predict_scan,
)
from boosting_qml.pqc.density_matrix import exact_noisy_reuploading_density_score

_EPS = np.finfo(np.float64).eps
_DTYPE = jnp.float32
_PARAM_SHIFT_COEFF_EPS = 1e-7


def _psi_g_with_sigma(alpha: jax.Array, *, sigma: float) -> jax.Array:
    sigma_arr = jnp.maximum(
        jnp.asarray(float(sigma), dtype=alpha.dtype),
        jnp.asarray(_EPS, dtype=alpha.dtype),
    )
    v = (1.0 - alpha) / sigma_arr
    cdf = 0.5 * (1.0 + erf(v / jnp.sqrt(2.0)))
    pdf = (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-(v**2) / 2.0)
    return cdf * (1.0 - alpha) + pdf * sigma_arr


def _psi_g_prime_with_sigma(alpha: jax.Array, *, sigma: float) -> jax.Array:
    sigma_arr = jnp.maximum(
        jnp.asarray(float(sigma), dtype=alpha.dtype),
        jnp.asarray(_EPS, dtype=alpha.dtype),
    )
    v = (1.0 - alpha) / sigma_arr
    cdf = 0.5 * (1.0 + erf(v / jnp.sqrt(2.0)))
    return -cdf


def _trainable_parameterized_gate_occurrence_count_for_config(
    *,
    n_qubits: int,
    n_layers: int,
    ansatz_kind: str,
    n_reuploads: int,
) -> int:
    if ansatz_kind == "legacy_reuploading":
        trainable_occurrences = int((4 * int(n_layers) + 3) * int(n_qubits))
        noisy_one_qubit_locations = int(
            legacy_reuploading_noisy_gate_counts(n_qubits=int(n_qubits), n_layers=int(n_layers))[
                "G1"
            ]
        )
        if trainable_occurrences != noisy_one_qubit_locations:
            raise AssertionError(
                "Current legacy parameter-shift accounting assumes every noisy one-qubit "
                "location is a shifted trainable gate occurrence. "
                f"trainable_occurrences={trainable_occurrences}, G1={noisy_one_qubit_locations}."
            )
        return trainable_occurrences
    if ansatz_kind == "paper_modular":
        return int(3 * int(n_reuploads) * int(n_layers) * int(n_qubits))
    raise ValueError(f"Unsupported ansatz_kind {ansatz_kind!r}.")


def _shifted_gate_occurrence_count_for_config(
    *,
    n_qubits: int,
    n_layers: int,
    ansatz_kind: str,
    n_reuploads: int,
) -> int:
    return _trainable_parameterized_gate_occurrence_count_for_config(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_kind=ansatz_kind,
        n_reuploads=n_reuploads,
    )


def _add_effective_shot_noise(
    score: jax.Array,
    key: jax.Array,
    n_shots_eff: int | None,
    *,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
) -> jax.Array:
    if n_shots_eff is None or int(n_shots_eff) <= 0:
        return score
    return sample_pauli_expectation_score(
        score,
        key,
        n_shots=int(n_shots_eff),
        distribution=measurement_sampling_distribution,
        gradient_mode=measurement_gradient_mode,
    )


def _predict_one_with_config(
    params: jax.Array,
    x: jax.Array,
    key_shot_noise: jax.Array,
    status_1q: Optional[jax.Array] = None,
    status_2q: Optional[jax.Array] = None,
    *,
    n_qubits: int,
    n_layers: int,
    n_shots_eff: int,
    ansatz_kind: str,
    n_reuploads: int,
    paper_data_chunk_mode: str,
    paper_observable: str,
    noise_backend: str,
    n_trajectories: int,
    trajectory_batch_size: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    mc_apply_two_qubit_noise: bool,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
) -> jax.Array:
    if noise_backend in {"mc_vmap", "mc_streaming"}:
        if ansatz_kind != "legacy_reuploading":
            raise ValueError(
                "Monte Carlo trajectory backends currently support only "
                "ansatz_kind='legacy_reuploading'."
            )
        if status_1q is not None and status_2q is not None:
            if noise_backend == "mc_vmap":
                if measurement_sampling_distribution == "binomial":
                    return noisy_reuploading_sampled_score_vmap_status(
                        key_shot_noise,
                        x,
                        n_qubits=int(n_qubits),
                        params=params,
                        n_layers=int(n_layers),
                        status_1q=status_1q,
                        status_2q=status_2q,
                        noise_strength_1q=float(noise_strength_1q),
                        noise_strength_2q=float(noise_strength_2q),
                        apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
                        observable="global_z",
                        measurement_gradient_mode=str(measurement_gradient_mode),
                    )
                score = noisy_reuploading_score_vmap_status(
                    x,
                    n_qubits=int(n_qubits),
                    params=params,
                    n_layers=int(n_layers),
                    status_1q=status_1q,
                    status_2q=status_2q,
                    noise_strength_1q=float(noise_strength_1q),
                    noise_strength_2q=float(noise_strength_2q),
                    apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
                    observable="global_z",
                )
            else:
                if measurement_sampling_distribution == "binomial":
                    return noisy_reuploading_sampled_score_streaming_status(
                        key_shot_noise,
                        x,
                        n_qubits=int(n_qubits),
                        params=params,
                        n_layers=int(n_layers),
                        status_1q=status_1q,
                        status_2q=status_2q,
                        noise_strength_1q=float(noise_strength_1q),
                        noise_strength_2q=float(noise_strength_2q),
                        trajectory_batch_size=int(trajectory_batch_size),
                        apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
                        observable="global_z",
                        measurement_gradient_mode=str(measurement_gradient_mode),
                    )
                score = noisy_reuploading_score_streaming_status(
                    x,
                    n_qubits=int(n_qubits),
                    params=params,
                    n_layers=int(n_layers),
                    status_1q=status_1q,
                    status_2q=status_2q,
                    noise_strength_1q=float(noise_strength_1q),
                    noise_strength_2q=float(noise_strength_2q),
                    trajectory_batch_size=int(trajectory_batch_size),
                    apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
                    observable="global_z",
                )
            return jnp.clip(
                _add_effective_shot_noise(
                    score,
                    key_shot_noise,
                    int(n_shots_eff),
                    measurement_sampling_distribution=str(measurement_sampling_distribution),
                    measurement_gradient_mode=str(measurement_gradient_mode),
                ),
                -1.0,
                1.0,
            )

        if noise_backend == "mc_vmap":
            return sample_noisy_reuploading_score_vmap(
                key_shot_noise,
                x,
                n_qubits=int(n_qubits),
                params=params,
                n_layers=int(n_layers),
                n_trajectories=int(n_trajectories),
                noise_strength_1q=float(noise_strength_1q),
                noise_strength_2q=float(noise_strength_2q),
                n_shots_eff=int(n_shots_eff),
                apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
                observable="global_z",
                measurement_sampling_distribution=str(measurement_sampling_distribution),
                measurement_gradient_mode=str(measurement_gradient_mode),
            )
        return sample_noisy_reuploading_score_streaming(
            key_shot_noise,
            x,
            n_qubits=int(n_qubits),
            params=params,
            n_layers=int(n_layers),
            n_trajectories=int(n_trajectories),
            noise_strength_1q=float(noise_strength_1q),
            noise_strength_2q=float(noise_strength_2q),
            n_shots_eff=int(n_shots_eff),
            trajectory_batch_size=int(trajectory_batch_size),
            apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
            observable="global_z",
            measurement_sampling_distribution=str(measurement_sampling_distribution),
            measurement_gradient_mode=str(measurement_gradient_mode),
        )

    if noise_backend == "density_matrix":
        if ansatz_kind != "legacy_reuploading":
            raise ValueError(
                "density_matrix noise_backend currently supports only "
                "ansatz_kind='legacy_reuploading'."
            )
        ypred = exact_noisy_reuploading_density_score(
            x,
            n_qubits=int(n_qubits),
            params=params,
            n_layers=int(n_layers),
            noise_strength_1q=float(noise_strength_1q),
            noise_strength_2q=float(noise_strength_2q),
            apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
            observable="global_z",
        )
        return jnp.clip(
            _add_effective_shot_noise(
                ypred,
                key_shot_noise,
                int(n_shots_eff),
                measurement_sampling_distribution=str(measurement_sampling_distribution),
                measurement_gradient_mode=str(measurement_gradient_mode),
            ),
            -1.0,
            1.0,
        )

    if ansatz_kind == "paper_modular":
        ypred = paper_modular_reuploading_predict_scan(
            X=x,
            n_qubits=int(n_qubits),
            params=params,
            n_layers=int(n_layers),
            n_reuploads=int(n_reuploads),
            chunk_mode=str(paper_data_chunk_mode),
            observable=str(paper_observable),
        )
    else:
        ypred = RX_RZ_RY_RZ_layer_scan(
            X=x,
            n_qubits=int(n_qubits),
            params=params,
            n_layers=int(n_layers),
        )

    if noise_backend == "noiseless":
        return jnp.clip(ypred, -1.0, 1.0)

    return jnp.clip(
        _add_effective_shot_noise(
            ypred,
            key_shot_noise,
            int(n_shots_eff),
            measurement_sampling_distribution=str(measurement_sampling_distribution),
            measurement_gradient_mode=str(measurement_gradient_mode),
        ),
        -1.0,
        1.0,
    )


def _predict_batch_with_config(
    params: jax.Array,
    x: jax.Array,
    key: jax.Array,
    status_1q: Optional[jax.Array] = None,
    status_2q: Optional[jax.Array] = None,
    **config: Any,
) -> jax.Array:
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(
        lambda x_i, k_i: _predict_one_with_config(
            params, x_i, k_i, status_1q=status_1q, status_2q=status_2q, **config
        )
    )(x, keys)


@lru_cache(maxsize=128)
def _cached_predict_batch(**config: Any):
    """Return a jitted, in-graph batch predictor keyed by circuit/noise config.

    Compiles once per distinct config and is reused across boosting rounds (each
    round builds a fresh learner with identical config). The returned function
    keeps its output as a device ``jax.Array`` so callers can chain further
    on-device work without a host round-trip (Opt 1: in-graph prediction).
    """

    @jax.jit
    def _predict(
        params: jax.Array,
        x: jax.Array,
        key: jax.Array,
        status_1q: Optional[jax.Array] = None,
        status_2q: Optional[jax.Array] = None,
    ) -> jax.Array:
        return _predict_batch_with_config(
            params, x, key, status_1q=status_1q, status_2q=status_2q, **config
        )

    return _predict


@lru_cache(maxsize=128)
def _cached_loss_fn(*, sigma: float, **config: Any):
    """Jitted weighted smooth-hinge loss, keyed by sigma + circuit/noise config.

    Used for the post-training full-dataset diagnostic so that final evaluation
    runs through a compiled (cached) kernel rather than an eager trace each round.
    """

    @jax.jit
    def _loss(
        params: jax.Array,
        x: jax.Array,
        y: jax.Array,
        w: jax.Array,
        key: jax.Array,
        status_1q: Optional[jax.Array] = None,
        status_2q: Optional[jax.Array] = None,
    ) -> jax.Array:
        return _loss_fn_with_config(
            params, x, y, w, key, status_1q=status_1q, status_2q=status_2q, sigma=float(sigma), **config
        )

    return _loss


def _loss_fn_with_config(
    params: jax.Array,
    xb: jax.Array,
    yb: jax.Array,
    wb: jax.Array,
    key: jax.Array,
    status_1q: Optional[jax.Array] = None,
    status_2q: Optional[jax.Array] = None,
    *,
    sigma: float,
    **config: Any,
) -> jax.Array:
    preds = _predict_batch_with_config(
        params, xb, key, status_1q=status_1q, status_2q=status_2q, **config
    )
    alpha = yb * preds
    per_example = _psi_g_with_sigma(alpha, sigma=float(sigma))
    wb = jnp.clip(wb, jnp.asarray(_EPS, dtype=wb.dtype))
    wb = wb / jnp.sum(wb)
    return jnp.sum(wb * per_example)


@lru_cache(maxsize=128)
def _cached_optimizer_and_train_step(
    *,
    learning_rate: float,
    grad_clip_norm: float,
    sigma: float,
    n_qubits: int,
    n_layers: int,
    n_shots_eff: int,
    ansatz_kind: str,
    n_reuploads: int,
    paper_data_chunk_mode: str,
    paper_observable: str,
    noise_backend: str,
    n_trajectories: int,
    trajectory_batch_size: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    mc_apply_two_qubit_noise: bool,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=float(grad_clip_norm)),
        optax.adam(learning_rate=float(learning_rate)),
    )
    config = dict(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots_eff=int(n_shots_eff),
        ansatz_kind=str(ansatz_kind),
        n_reuploads=int(n_reuploads),
        paper_data_chunk_mode=str(paper_data_chunk_mode),
        paper_observable=str(paper_observable),
        noise_backend=str(noise_backend),
        n_trajectories=int(n_trajectories),
        trajectory_batch_size=int(trajectory_batch_size),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        mc_apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
        measurement_sampling_distribution=str(measurement_sampling_distribution),
        measurement_gradient_mode=str(measurement_gradient_mode),
    )

    @jax.jit
    def _train_step(
        params_local: jax.Array,
        opt_state_local: optax.OptState,
        xb: jax.Array,
        yb: jax.Array,
        wb: jax.Array,
        key_local: jax.Array,
        status_1q: Optional[jax.Array] = None,
        status_2q: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, optax.OptState, jax.Array]:
        loss_value, grads = jax.value_and_grad(_loss_fn_with_config)(
            params_local,
            xb,
            yb,
            wb,
            key_local,
            status_1q=status_1q,
            status_2q=status_2q,
            sigma=float(sigma),
            **config,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state_local, params_local)
        new_params = optax.apply_updates(params_local, updates)
        return new_params, new_opt_state, loss_value

    return optimizer, _train_step


@lru_cache(maxsize=64)
def _cached_parameter_shift_optimizer_and_train_step(
    *,
    learning_rate: float,
    grad_clip_norm: float,
    sigma: float,
    n_qubits: int,
    n_layers: int,
    n_shots_eff: int,
    ansatz_kind: str,
    n_reuploads: int,
    paper_data_chunk_mode: str,
    paper_observable: str,
    noise_backend: str,
    n_trajectories: int,
    trajectory_batch_size: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    mc_apply_two_qubit_noise: bool,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
    parameter_shift_chunk_size: int | str | None,
):
    if str(measurement_gradient_mode) != "parameter_shift_binomial":
        raise ValueError(
            "The parameter-shift trainer requires "
            "measurement_gradient_mode='parameter_shift_binomial'."
        )
    if str(measurement_sampling_distribution) != "binomial":
        raise ValueError(
            "The parameter-shift trainer requires "
            "measurement_sampling_distribution='binomial'."
        )
    if str(ansatz_kind) != "legacy_reuploading":
        raise ValueError(
            "The finite-shot parameter-shift trainer currently supports only "
            "ansatz_kind='legacy_reuploading'."
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=float(grad_clip_norm)),
        optax.adam(learning_rate=float(learning_rate)),
    )
    config = dict(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots_eff=int(n_shots_eff),
        ansatz_kind=str(ansatz_kind),
        n_reuploads=int(n_reuploads),
        paper_data_chunk_mode=str(paper_data_chunk_mode),
        paper_observable=str(paper_observable),
        noise_backend=str(noise_backend),
        n_trajectories=int(n_trajectories),
        trajectory_batch_size=int(trajectory_batch_size),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        mc_apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
        measurement_sampling_distribution=str(measurement_sampling_distribution),
        measurement_gradient_mode=str(measurement_gradient_mode),
    )
    def _finite_shot_plugin_loss_and_grads(
        params_local: jax.Array,
        xb: jax.Array,
        yb: jax.Array,
        wb: jax.Array,
        key_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        grads, loss_value, _ = finite_shot_plugin_loss_grads_and_preds_density(
            params_local,
            xb,
            yb,
            key_local,
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
            n_shots=int(n_shots_eff),
            noise_strength_1q=float(noise_strength_1q),
            noise_strength_2q=float(noise_strength_2q),
            apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
            weights=wb,
            sigma=float(sigma),
            ansatz_kind=str(ansatz_kind),
            noise_backend=str(noise_backend),
            measurement_sampling_distribution=str(measurement_sampling_distribution),
            measurement_gradient_mode=str(measurement_gradient_mode),
            observable="global_z",
            parameter_shift_chunk_size=parameter_shift_chunk_size,
        )
        return loss_value, grads

    def _train_step(
        params_local: jax.Array,
        opt_state_local: optax.OptState,
        xb: jax.Array,
        yb: jax.Array,
        wb: jax.Array,
        key_local: jax.Array,
        status_1q: Optional[jax.Array] = None,
        status_2q: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, optax.OptState, jax.Array]:
        del status_1q, status_2q
        loss_value, grads = _finite_shot_plugin_loss_and_grads(
            params_local,
            xb,
            yb,
            wb,
            key_local,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state_local, params_local)
        new_params = optax.apply_updates(params_local, updates)
        return new_params, new_opt_state, loss_value

    return optimizer, _train_step


@lru_cache(maxsize=64)
def _cached_scanned_trainer(
    *,
    learning_rate: float,
    grad_clip_norm: float,
    sigma: float,
    n_qubits: int,
    n_layers: int,
    n_shots_eff: int,
    ansatz_kind: str,
    n_reuploads: int,
    paper_data_chunk_mode: str,
    paper_observable: str,
    noise_backend: str,
    n_trajectories: int,
    trajectory_batch_size: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    mc_apply_two_qubit_noise: bool,
    measurement_sampling_distribution: str,
    measurement_gradient_mode: str,
):
    """Return ``(optimizer, train_all)`` where ``train_all`` runs a fixed
    epoch x minibatch chunk as a single scanned, jitted graph (Opt 2).

    This mirrors the prototype's execution model: one host dispatch trains the
    requested chunk, minimizing per-minibatch Python/dispatch overhead in the
    small-model regime. ``perm`` has shape ``(chunk_epochs, n_batches, batch_size)``
    so all batch shapes are static; the remainder ``n_samples % batch_size`` is
    dropped (the caller documents this). Adaptive training repeatedly calls the
    same chunk shape, so XLA can reuse the compiled kernel.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=float(grad_clip_norm)),
        optax.adam(learning_rate=float(learning_rate)),
    )
    config = dict(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        n_shots_eff=int(n_shots_eff),
        ansatz_kind=str(ansatz_kind),
        n_reuploads=int(n_reuploads),
        paper_data_chunk_mode=str(paper_data_chunk_mode),
        paper_observable=str(paper_observable),
        noise_backend=str(noise_backend),
        n_trajectories=int(n_trajectories),
        trajectory_batch_size=int(trajectory_batch_size),
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        mc_apply_two_qubit_noise=bool(mc_apply_two_qubit_noise),
        measurement_sampling_distribution=str(measurement_sampling_distribution),
        measurement_gradient_mode=str(measurement_gradient_mode),
    )

    @jax.jit
    def _train_all(
        params: jax.Array,
        opt_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
        w: jax.Array,
        perm: jax.Array,
        key: jax.Array,
        status_1q: Optional[jax.Array] = None,
        status_2q: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, optax.OptState, jax.Array]:
        def batch_body(carry, idx_row):
            params_c, opt_state_c, key_c = carry
            key_c, key_b = jax.random.split(key_c)
            xb = x[idx_row]
            yb = y[idx_row]
            wb = w[idx_row]
            loss_value, grads = jax.value_and_grad(_loss_fn_with_config)(
                params_c,
                xb,
                yb,
                wb,
                key_b,
                status_1q=status_1q,
                status_2q=status_2q,
                sigma=float(sigma),
                **config,
            )
            updates, opt_state_c = optimizer.update(grads, opt_state_c, params_c)
            params_c = optax.apply_updates(params_c, updates)
            return (params_c, opt_state_c, key_c), loss_value

        def epoch_body(carry, epoch_perm):
            carry, batch_losses = jax.lax.scan(batch_body, carry, epoch_perm)
            return carry, jnp.mean(batch_losses)

        (params, opt_state, key), epoch_losses = jax.lax.scan(epoch_body, (params, opt_state, key), perm)
        return params, opt_state, epoch_losses

    return optimizer, _train_all


@dataclass
class WeightedPQCBinaryWeakLearner(BinaryWeakLearner):
    """Weighted smooth-hinge PQC weak learner for Real AdaBoost."""

    n_qubits: int
    n_layers: int
    n_shots_eff: int = int(1e4)
    n_epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-2
    grad_clip_norm: float = 1.0
    sigma: float = 0.5
    scan_training: bool = False  # Opt 2: run fixed epoch x minibatch chunks as one scanned graph
    adaptive_epoch_stopping: bool = False
    adaptive_epoch_window_size: int = 50
    adaptive_epoch_min_epochs: int = 100
    adaptive_epoch_patience: int = 2
    adaptive_epoch_scale: float = 0.75
    adaptive_epoch_chunk_size: int = 10
    training_stopping_definition: str = BEST_LOSS_NO_IMPROVEMENT_DEFINITION
    fixed_min_epochs: Optional[int] = None
    early_stopping_enabled: bool = True
    early_stopping_patience_epochs: int = 20
    early_stopping_check_frequency_epochs: int = 1
    early_stopping_min_delta: float = 0.0
    restore_best_parameters: bool = True
    updates_per_epoch: Optional[int] = None
    monitored_loss: str = (
        "adaboost_weighted_epoch_average_smooth_hinge_PsiG_sigma_0p5"
    )
    intra_round_target_stopping: bool = True
    target_check_frequency_epochs: int = 10
    check_on_training_stop: bool = True
    training_accuracy_target: Optional[float] = None

    ansatz_kind: str = "legacy_reuploading"
    n_reuploads: int = 1
    paper_data_chunk_mode: str = "cyclic"
    paper_observable: str = "z0"

    noise_backend: str = "effective_shot"
    n_trajectories: int = 1
    trajectory_batch_size: int = 16
    noise_strength_1q: float = 0.0
    noise_strength_2q: float = 0.0
    mc_apply_two_qubit_noise: bool = True
    # Main physical runs use fresh trajectory tables for every expectation-value
    # evaluation. ``fixed`` is retained for explicit sample-average ablations.
    mc_status_mode: str = "fresh"
    measurement_sampling_distribution: str = "binomial"
    measurement_gradient_mode: str = "straight_through_expectation"
    parameter_shift_chunk_size: int | str | None = 0
    compute_fit_diagnostic: bool = True
    raw_shot_overhead: float = 1.0
    verbose: bool = False
    log_every_epochs: int = 50

    param_init_scheme: str = "block_normal"
    param_init_std: float = 1.0
    param_init_encoding_mean: float = 1.0
    param_init_encoding_std: float = 0.1
    param_init_variational_mean: float = 0.0
    param_init_variational_std: float = 0.1
    param_init_final_mean: float = 0.0
    param_init_final_std: float = 0.1
    param_init_teacher_mean: Optional[Array] = None
    param_init_teacher_std: float = 0.1
    param_init_clip_abs: Optional[float] = None

    params_: Optional[jax.Array] = field(default=None, init=False, repr=False)
    accepts_jax_example_weights: bool = field(default=True, init=False, repr=False)
    supports_jax_prediction: bool = field(default=True, init=False, repr=False)
    _train_model_evaluations: int = field(default=0, init=False, repr=False)
    _inference_model_evaluations: int = field(default=0, init=False, repr=False)
    _unshifted_loss_evaluations: int = field(default=0, init=False, repr=False)
    _shifted_gradient_evaluations: int = field(default=0, init=False, repr=False)
    _training_monitor_evaluations: int = field(default=0, init=False, repr=False)
    _prediction_phase: str = field(default="inference", init=False, repr=False)
    _mc_status_1q_: Optional[jax.Array] = field(default=None, init=False, repr=False)
    _mc_status_2q_: Optional[jax.Array] = field(default=None, init=False, repr=False)
    training_resume_state_: Optional[Mapping[str, Any]] = field(
        default=None, init=False, repr=False
    )
    last_training_stopping_: Optional[Mapping[str, Any]] = field(
        default=None, init=False, repr=False
    )
    optimizer_updates_: int = field(default=0, init=False, repr=False)

    def reset_shot_accounting(self) -> None:
        """Reset mutable train/inference shot counters for one boosting round."""

        self._train_model_evaluations = 0
        self._inference_model_evaluations = 0
        self._unshifted_loss_evaluations = 0
        self._shifted_gradient_evaluations = 0
        self._training_monitor_evaluations = 0

    @contextmanager
    def prediction_phase(self, phase: str):
        """Temporarily attribute ``predict_score`` evaluations to a budget phase.

        The boosting loop wraps its per-round diagnostic scoring (train/val/test
        ensemble evaluations) in ``prediction_phase("train")`` so those passes are
        accounted as training cost rather than inference cost.
        """

        if phase not in {"train", "inference"}:
            raise ValueError(f"phase must be 'train' or 'inference', got {phase!r}.")
        previous = self._prediction_phase
        self._prediction_phase = phase
        try:
            yield
        finally:
            self._prediction_phase = previous


    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[PQC weak learner] {message}", flush=True)

    def _should_log_epoch(self, epoch: int, *, converged: bool = False) -> bool:
        if not self.verbose:
            return False
        every = max(1, int(self.log_every_epochs))
        return bool(converged or epoch == 1 or epoch >= int(self.n_epochs) or epoch % every == 0)

    def _adaptive_epoch_config(self) -> WindowedPlateauConfig:
        return WindowedPlateauConfig(
            enabled=bool(self.adaptive_epoch_stopping),
            window_size=int(self.adaptive_epoch_window_size),
            min_steps=int(self.adaptive_epoch_min_epochs),
            patience=int(self.adaptive_epoch_patience),
            scale=float(self.adaptive_epoch_scale),
        )

    def _adaptive_epoch_chunk_size(self) -> int:
        chunk_size = int(self.adaptive_epoch_chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"adaptive_epoch_chunk_size must be positive, got {chunk_size}.")
        return chunk_size

    def _uses_mc_backend(self) -> bool:
        return self.noise_backend in {"mc_vmap", "mc_streaming"}

    def _validate_mc_status_mode(self) -> None:
        valid_modes = {"fresh", "fixed"}
        if self.mc_status_mode not in valid_modes:
            raise ValueError(
                f"mc_status_mode must be one of {sorted(valid_modes)}, got {self.mc_status_mode!r}."
            )
        # The status convention only affects explicit Monte Carlo trajectory
        # backends. For effective_shot runs there is no trajectory table to
        # reuse, so the default ``fixed`` mode is harmless and ignored.
        if self.mc_status_mode == "fixed" and self._uses_mc_backend() and self.ansatz_kind != "legacy_reuploading":
            raise ValueError("mc_status_mode='fixed' currently supports only ansatz_kind='legacy_reuploading'.")

    def _initialize_mc_statuses(self, key: jax.Array) -> None:
        if self.mc_status_mode == "fixed" and self._uses_mc_backend():
            self._mc_status_1q_, self._mc_status_2q_ = make_reuploading_mc_statuses(
                key,
                n_trajectories=int(self.n_trajectories),
                n_qubits=int(self.n_qubits),
                n_layers=int(self.n_layers),
                dtype=_DTYPE,
            )
        else:
            self._mc_status_1q_ = None
            self._mc_status_2q_ = None

    def _mc_status_args(self) -> tuple[Optional[jax.Array], Optional[jax.Array]]:
        if self.mc_status_mode == "fixed" and self._uses_mc_backend():
            if self._mc_status_1q_ is None or self._mc_status_2q_ is None:
                raise RuntimeError("Fixed MC statuses are not initialized. Call fit_weighted first.")
            return self._mc_status_1q_, self._mc_status_2q_
        return None, None

    def _trajectory_multiplier(self) -> int:
        if self.noise_backend in {"mc_vmap", "mc_streaming"}:
            return int(self.n_trajectories)
        return 1

    def _shots_per_model_evaluation(self) -> int:
        if self._uses_mc_backend() and self.measurement_sampling_distribution == "binomial":
            return 1
        return int(self.n_shots_eff)

    def _uses_parameter_shift_gradient(self) -> bool:
        return str(self.measurement_gradient_mode) == "parameter_shift_binomial"

    def _shifted_parameter_count(self) -> int:
        return _shifted_gate_occurrence_count_for_config(
            n_qubits=int(self.n_qubits),
            n_layers=int(self.n_layers),
            ansatz_kind=str(self.ansatz_kind),
            n_reuploads=int(self.n_reuploads),
        )

    def _shifted_gate_occurrence_count(self) -> int:
        return self._shifted_parameter_count()

    def _gradient_loss_estimator(self) -> str:
        if self._uses_parameter_shift_gradient():
            return "finite_shot_plugin"
        if self.measurement_gradient_mode == "straight_through_expectation":
            return "autodiff_straight_through"
        return "sample_stop_gradient"

    def _parameter_shift_noise_coupling(self) -> str:
        if self._uses_parameter_shift_gradient():
            return "independent"
        return "not_applicable_autodiff_straight_through"

    def _mc_refresh_granularity(self) -> str:
        if self.noise_backend == "density_matrix":
            return "not_applicable_exact_channel"
        if not self._uses_mc_backend():
            return "not_applicable"
        if self.mc_status_mode == "fresh":
            return "per_expectation_value_evaluation"
        return "fixed_per_weak_learner"

    def _physical_noise_coupling(self) -> str:
        if self.noise_backend == "density_matrix":
            return "exact_mixed_state_channel_no_mc"
        if not self._uses_mc_backend():
            return "not_applicable"
        if self.mc_status_mode == "fresh":
            return "independent_per_circuit_evaluation"
        return "fixed_trajectory_table_per_weak_learner"

    def shot_accounting_summary(self) -> dict[str, Any]:
        """Return train/inference effective and raw shot accounting."""

        summary = build_shot_accounting_summary(
            n_shots_eff_per_expectation=int(self._shots_per_model_evaluation()),
            train_model_evaluations=int(self._train_model_evaluations),
            inference_model_evaluations=int(self._inference_model_evaluations),
            raw_shot_overhead=float(self.raw_shot_overhead),
            trajectory_multiplier=self._trajectory_multiplier(),
            trajectory_multiplier_counts_as_shots=bool(
                self._uses_mc_backend() and self.measurement_sampling_distribution == "binomial"
            ),
            source=f"weighted_pqc_binary:{self.noise_backend}:{self.measurement_sampling_distribution}",
        ).to_dict()
        shots_per_expectation = int(
            summary["n_shots_eff_per_expectation"] * summary["trajectory_multiplier"]
        )
        summary.update(
            {
                "configured_n_shots_eff": int(self.n_shots_eff),
                "shots_per_expectation": shots_per_expectation,
                "measurement_sampling_distribution": str(self.measurement_sampling_distribution),
                "measurement_gradient_mode": str(self.measurement_gradient_mode),
                "parameter_shift_chunk_size": self.parameter_shift_chunk_size,
                "gradient_loss_estimator": self._gradient_loss_estimator(),
                "gradient_unshifted_shots": float(
                    int(self._unshifted_loss_evaluations)
                    * shots_per_expectation
                    * float(self.raw_shot_overhead)
                ),
                "gradient_shifted_shots": float(
                    int(self._shifted_gradient_evaluations)
                    * shots_per_expectation
                    * float(self.raw_shot_overhead)
                ),
                "shifted_parameter_count": int(self._shifted_parameter_count()),
                "shifted_gate_occurrence_count": int(self._shifted_gate_occurrence_count()),
                "unshifted_loss_evaluations": int(self._unshifted_loss_evaluations),
                "shifted_gradient_evaluations": int(self._shifted_gradient_evaluations),
                "training_monitor_evaluations": int(self._training_monitor_evaluations),
                "optimizer_updates": int(self.optimizer_updates_),
                "gradient_evaluations": int(self.optimizer_updates_),
                "mc_status_mode": str(self.mc_status_mode) if self._uses_mc_backend() else "not_applicable",
                "mc_refresh_granularity": self._mc_refresh_granularity(),
                "mc_trajectories_per_expectation": (
                    int(self.n_trajectories) if self._uses_mc_backend() else 0
                ),
                "physical_noise_coupling": self._physical_noise_coupling(),
                "parameter_shift_noise_coupling": self._parameter_shift_noise_coupling(),
                "double_stochastic_simulation": bool(
                    self._uses_mc_backend()
                    and self.measurement_sampling_distribution != "binomial"
                    and int(self.n_trajectories) > 1
                    and int(self.n_shots_eff) > 1
                ),
            }
        )
        return summary

    def _check_feature_alignment(self, n_features: int) -> None:
        if self.ansatz_kind == "legacy_reuploading" and n_features != int(self.n_qubits):
            raise ValueError(
                "For ansatz_kind='legacy_reuploading', dataset.n_features must equal n_qubits "
                f"(got n_features={n_features}, n_qubits={self.n_qubits})."
            )

        if self.ansatz_kind == "paper_modular" and self.paper_data_chunk_mode == "strict":
            required = int(3 * self.n_qubits * self.n_layers)
            if n_features != required:
                raise ValueError(
                    "For ansatz_kind='paper_modular' with paper_data_chunk_mode='strict', "
                    f"dataset.n_features must be {required}, got {n_features}."
                )

    def _initialize_params(self, key: jax.Array) -> jax.Array:
        teacher_mean = None
        if self.param_init_teacher_mean is not None:
            teacher_mean = jnp.asarray(self.param_init_teacher_mean, dtype=_DTYPE)

        if self.ansatz_kind == "paper_modular":
            return initialize_paper_modular_params(
                key=key,
                n_qubits=int(self.n_qubits),
                n_layers=int(self.n_layers),
                n_reuploads=int(self.n_reuploads),
                scheme=str(self.param_init_scheme),
                std=float(self.param_init_std),
                variational_mean=float(self.param_init_variational_mean),
                variational_std=float(self.param_init_variational_std),
                teacher_params_mean=teacher_mean,
                teacher_std=float(self.param_init_teacher_std),
                clip_abs=self.param_init_clip_abs,
            )

        return initialize_reuploading_params(
            key=key,
            n_qubits=int(self.n_qubits),
            n_layers=int(self.n_layers),
            scheme=str(self.param_init_scheme),
            std=float(self.param_init_std),
            encoding_mean=float(self.param_init_encoding_mean),
            encoding_std=float(self.param_init_encoding_std),
            variational_mean=float(self.param_init_variational_mean),
            variational_std=float(self.param_init_variational_std),
            final_mean=float(self.param_init_final_mean),
            final_std=float(self.param_init_final_std),
            teacher_params_mean=teacher_mean,
            teacher_std=float(self.param_init_teacher_std),
            clip_abs=self.param_init_clip_abs,
        )

    def _predict_config(self) -> dict[str, Any]:
        # These fields are treated as immutable after learner construction. If a
        # circuit/noise setting changes, create a new learner so JAX retraces.
        return dict(
            n_qubits=int(self.n_qubits),
            n_layers=int(self.n_layers),
            n_shots_eff=int(self.n_shots_eff),
            ansatz_kind=str(self.ansatz_kind),
            n_reuploads=int(self.n_reuploads),
            paper_data_chunk_mode=str(self.paper_data_chunk_mode),
            paper_observable=str(self.paper_observable),
            noise_backend=str(self.noise_backend),
            n_trajectories=int(self.n_trajectories),
            trajectory_batch_size=int(self.trajectory_batch_size),
            noise_strength_1q=float(self.noise_strength_1q),
            noise_strength_2q=float(self.noise_strength_2q),
            mc_apply_two_qubit_noise=bool(self.mc_apply_two_qubit_noise),
            measurement_sampling_distribution=str(self.measurement_sampling_distribution),
            measurement_gradient_mode=str(self.measurement_gradient_mode),
        )

    def _predict_one(self, params: jax.Array, x: jax.Array, key_shot_noise: jax.Array) -> jax.Array:
        status_1q, status_2q = self._mc_status_args()
        return _predict_one_with_config(
            params, x, key_shot_noise, status_1q=status_1q, status_2q=status_2q, **self._predict_config()
        )

    def _predict_batch(self, params: jax.Array, x: jax.Array, key: jax.Array) -> jax.Array:
        status_1q, status_2q = self._mc_status_args()
        return _predict_batch_with_config(
            params, x, key, status_1q=status_1q, status_2q=status_2q, **self._predict_config()
        )

    def _psi_g(self, alpha: jax.Array) -> jax.Array:
        return _psi_g_with_sigma(alpha, sigma=float(self.sigma))

    def fit_weighted(
        self,
        dataset: BinaryDataset,
        example_weights: Array,
        rng: np.random.Generator,
        epoch_callback: Optional[
            Callable[[Any, int, str], Mapping[str, Any] | None]
        ] = None,
    ) -> WeakLearnerFitResult:
        if self.n_shots_eff <= 0:
            raise ValueError(f"n_shots_eff must be positive, got {self.n_shots_eff}.")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}.")
        if self.training_stopping_definition != BEST_LOSS_NO_IMPROVEMENT_DEFINITION:
            raise ValueError(
                f"Unsupported training_stopping_definition={self.training_stopping_definition!r}."
            )
        if self.fixed_min_epochs not in {None, 0}:
            raise ValueError("The final protocol has no fixed minimum epoch count.")
        stopping_config = BestLossNoImprovementConfig(
            enabled=bool(self.early_stopping_enabled),
            patience=int(self.early_stopping_patience_epochs),
            min_delta=float(self.early_stopping_min_delta),
            check_frequency_epochs=int(
                self.early_stopping_check_frequency_epochs
            ),
        )
        stopping_config.validate()
        if not self.restore_best_parameters:
            raise ValueError("The final protocol requires restore_best_parameters=true.")
        if (
            self.monitored_loss
            != "adaboost_weighted_epoch_average_smooth_hinge_PsiG_sigma_0p5"
        ):
            raise ValueError(
                "The final weak-PQC protocol monitors weighted epoch-average "
                "smooth-hinge loss."
            )
        expected_updates_per_epoch = int(
            np.ceil(dataset.n_samples / int(self.batch_size))
        )
        if (
            self.updates_per_epoch is not None
            and int(self.updates_per_epoch) != expected_updates_per_epoch
        ):
            raise ValueError(
                "updates_per_epoch disagrees with the actual dataset and batch size."
            )
        if int(self.target_check_frequency_epochs) != 10:
            raise ValueError("The final intra-round target-check cadence is 10 epochs.")
        if (
            self.training_accuracy_target is not None
            and self.intra_round_target_stopping
        ):
            raise ValueError(
                "Standalone weak-learner accuracy cannot trigger final-protocol stopping."
            )
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        valid_backends = {"density_matrix", "effective_shot", "mc_vmap", "mc_streaming", "noiseless"}
        if self.noise_backend not in valid_backends:
            raise ValueError(f"noise_backend must be one of {sorted(valid_backends)}, got {self.noise_backend!r}.")
        if self.n_trajectories <= 0:
            raise ValueError(f"n_trajectories must be positive, got {self.n_trajectories}.")
        if self.trajectory_batch_size <= 0:
            raise ValueError(f"trajectory_batch_size must be positive, got {self.trajectory_batch_size}.")
        self._validate_mc_status_mode()
        self.measurement_sampling_distribution = validate_measurement_sampling_distribution(
            self.measurement_sampling_distribution
        )
        self.measurement_gradient_mode = validate_measurement_gradient_mode(self.measurement_gradient_mode)
        if self._uses_parameter_shift_gradient():
            if self.measurement_sampling_distribution != "binomial":
                raise ValueError(
                    "measurement_gradient_mode='parameter_shift_binomial' requires "
                    "measurement_sampling_distribution='binomial'."
                )
            if self.ansatz_kind != "legacy_reuploading":
                raise ValueError(
                    "measurement_gradient_mode='parameter_shift_binomial' currently supports "
                    "only ansatz_kind='legacy_reuploading'."
                )
            if self._uses_mc_backend() and self.mc_status_mode != "fresh":
                raise ValueError(
                    "measurement_gradient_mode='parameter_shift_binomial' requires "
                    "mc_status_mode='fresh' for independent physical-noise samples."
                )
        validate_integer_shots(int(self.n_shots_eff))
        if self.raw_shot_overhead < 1.0:
            raise ValueError(f"raw_shot_overhead must be >= 1, got {self.raw_shot_overhead}.")

        resume_state = (
            None
            if self.training_resume_state_ is None
            else dict(self.training_resume_state_)
        )
        self.reset_shot_accounting()
        if resume_state is not None:
            ledger = dict(resume_state["resource_ledger"])
            self._unshifted_loss_evaluations = int(
                ledger.get("unshifted_loss_evaluations", 0)
            )
            self._shifted_gradient_evaluations = int(
                ledger.get(
                    "shifted_gradient_model_evaluations",
                    ledger.get("gradient_evaluations", 0),
                )
            )
            self._training_monitor_evaluations = int(
                ledger.get("training_monitor_evaluations", 0)
            )
        expected_unshifted_loss_evaluations = int(
            self._unshifted_loss_evaluations
        )

        self._check_feature_alignment(dataset.n_features)

        x = jnp.asarray(dataset.X, dtype=_DTYPE)
        y = jnp.asarray(dataset.y, dtype=_DTYPE)
        w = jnp.asarray(example_weights, dtype=_DTYPE)
        if tuple(w.shape) != (dataset.n_samples,):
            raise ValueError("example_weights must have shape [n_samples].")

        w = jnp.clip(w, jnp.asarray(_EPS, dtype=_DTYPE))
        w = w / jnp.sum(w)

        seed = (
            int(rng.integers(0, np.iinfo(np.int32).max))
            if resume_state is None
            else int(resume_state["fit_seed"])
        )

        def fit_key(*parts: Any) -> jax.Array:
            return jax.random.PRNGKey(weighted_pqc_fit_seed(seed, *parts))

        def fit_np_rng(*parts: Any) -> np.random.Generator:
            return np.random.default_rng(weighted_pqc_fit_seed(seed, *parts))

        fit_t0 = time.perf_counter()
        params = (
            self._initialize_params(fit_key("init"))
            if resume_state is None
            else jnp.asarray(resume_state["current_parameters"], dtype=_DTYPE)
        )
        if resume_state is None or self._mc_status_1q_ is None:
            self._initialize_mc_statuses(fit_key("physical_status"))
        status_1q, status_2q = self._mc_status_args()
        convergence_tracker = (
            BestLossNoImprovementTracker(stopping_config)
            if resume_state is None
            else BestLossNoImprovementTracker.from_state_dict(
                dict(resume_state["stopping_state"])
            )
        )
        epoch_loss_history: list[float] = []
        epochs_run = int(convergence_tracker.current_epoch)
        optimizer_updates = int(
            0
            if resume_state is None
            else resume_state["resource_ledger"]["optimizer_updates"]
        )
        best_params = (
            None
            if resume_state is None or resume_state.get("best_parameters") is None
            else jnp.asarray(resume_state["best_parameters"], dtype=_DTYPE)
        )
        best_opt_state = None
        best_rng_state = (
            None
            if resume_state is None or resume_state.get("best_rng_state") is None
            else jnp.asarray(resume_state["best_rng_state"])
        )
        current_rng_state = (
            None
            if resume_state is None
            else resume_state.get("rng_state")
        )
        final_epoch_loss_before_restoration = None
        target_confirmed = False
        target_confirmation: Mapping[str, Any] | None = None
        target_check_history: list[dict[str, Any]] = []
        training_stop_reason = f"reached_max_epochs_{int(self.n_epochs)}"

        def _restore_optimizer_state(initial_state: Any) -> Any:
            if resume_state is None:
                return initial_state
            leaves = [
                jnp.asarray(value)
                for value in resume_state["current_optimizer_state_leaves"]
            ]
            treedef = jax.tree_util.tree_structure(initial_state)
            if len(leaves) != treedef.num_leaves:
                raise ValueError("Optimizer resume state has an incompatible structure.")
            return jax.tree_util.tree_unflatten(treedef, leaves)

        def _restore_best_optimizer_state(initial_state: Any) -> Any:
            if resume_state is None or resume_state.get("best_optimizer_state_leaves") is None:
                return None
            leaves = [
                jnp.asarray(value)
                for value in resume_state["best_optimizer_state_leaves"]
            ]
            treedef = jax.tree_util.tree_structure(initial_state)
            if len(leaves) != treedef.num_leaves:
                raise ValueError("Best optimizer resume state has an incompatible structure.")
            return jax.tree_util.tree_unflatten(treedef, leaves)

        def _record_completed_epoch(
            *,
            epoch_loss: float,
            current_params: jax.Array,
            current_opt_state: Any,
            epoch_rng_state: Any,
            at_max_epochs: bool,
        ):
            nonlocal best_params, best_opt_state, best_rng_state, current_rng_state
            nonlocal target_confirmed, target_confirmation, training_stop_reason
            nonlocal train_model_evaluations
            decision = convergence_tracker.update(epoch_loss, epoch=epochs_run)
            current_rng_state = jnp.array(epoch_rng_state)
            if decision.improved:
                best_params = jnp.array(current_params)
                best_opt_state = jax.tree_util.tree_map(
                    lambda value: jnp.array(value),
                    current_opt_state,
                )
                best_rng_state = jnp.array(epoch_rng_state)

            loss_stop = bool(decision.should_stop)
            check_reason = None
            if epochs_run % int(self.target_check_frequency_epochs) == 0:
                check_reason = "cadence"
            if loss_stop and bool(self.check_on_training_stop):
                check_reason = "loss_based_training_stop"
            if at_max_epochs:
                check_reason = "max_epochs"

            if (
                epoch_callback is not None
                and bool(self.intra_round_target_stopping)
                and check_reason is not None
            ):
                self.params_ = jnp.array(current_params)
                callback_train_evaluations_before = int(
                    self._train_model_evaluations
                )
                callback_result = epoch_callback(self, int(epochs_run), check_reason)
                callback_train_evaluations = (
                    int(self._train_model_evaluations)
                    - callback_train_evaluations_before
                )
                if callback_train_evaluations < 0:
                    raise AssertionError(
                        "The epoch callback decreased the weak learner's "
                        "training model-evaluation ledger."
                    )
                # predict_score updates the learner-owned counter, whereas the
                # optimizer loop uses this local counter. Preserve callback
                # evaluations before the local total is assigned back at fit end.
                train_model_evaluations += callback_train_evaluations
                callback_payload = dict(callback_result or {})
                callback_payload.update(
                    {
                        "epoch": int(epochs_run),
                        "check_reason": str(check_reason),
                    }
                )
                target_check_history.append(callback_payload)
                if bool(callback_payload.get("target_confirmed", False)):
                    target_confirmed = True
                    target_confirmation = callback_payload
                    training_stop_reason = "target_confirmed_during_weak_training"

            if loss_stop and not target_confirmed:
                training_stop_reason = (
                    f"no_improvement_patience_{int(self.early_stopping_patience_epochs)}"
                )
            return decision

        n_samples = dataset.n_samples
        batch_size = int(self.batch_size)
        n_full_batches = n_samples // batch_size
        self._log(
            "fit start "
            f"samples={n_samples} features={dataset.n_features} epochs_max={int(self.n_epochs)} "
            f"batch_size={batch_size} scan={bool(self.scan_training)} "
            f"adaptive={bool(self.adaptive_epoch_stopping)} backend={self.noise_backend} "
            f"trajectories={int(self.n_trajectories)} status={self.mc_status_mode} "
            f"measurement={self.measurement_sampling_distribution}"
        )
        # Opt 2: scan fixed epoch x minibatch chunks as one graph when
        # requested and at least one full minibatch exists. Falls back to the
        # per-minibatch Python loop otherwise (e.g. n_samples < batch_size).
        if self._uses_parameter_shift_gradient() and self.scan_training:
            warnings.warn(
                "scan_training=True is ignored for parameter_shift_binomial because the "
                "scanned trainer is autodiff-only.",
                RuntimeWarning,
                stacklevel=2,
            )
        use_scan = (
            bool(self.scan_training)
            and n_full_batches >= 1
            and not self._uses_parameter_shift_gradient()
        )

        if use_scan:
            remainder = n_samples % batch_size
            if remainder != 0:
                warnings.warn(
                    f"scan_training=True drops the final partial batch: {remainder} of "
                    f"{n_samples} examples are excluded every epoch (batch_size={batch_size}). "
                    "Use a train size divisible by batch_size for full-data training, or set "
                    "scan_training=False for science runs.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            optimizer, train_all = _cached_scanned_trainer(
                learning_rate=float(self.learning_rate),
                grad_clip_norm=float(self.grad_clip_norm),
                sigma=float(self.sigma),
                **self._predict_config(),
            )
            initial_opt_state = optimizer.init(params)
            opt_state = _restore_optimizer_state(initial_opt_state)
            best_opt_state = _restore_best_optimizer_state(initial_opt_state)
            train_model_evaluations = int(
                0
                if resume_state is None
                else resume_state["resource_ledger"]["training_model_evaluations"]
            )

            # A single-epoch scan retains compiled minibatch execution while
            # exposing parameters after every epoch for patience and checkpointing.
            chunk_sizes = [1] * (int(self.n_epochs) - epochs_run)

            for chunk_idx, chunk_epochs in enumerate(chunk_sizes, start=1):
                chunk_t0 = time.perf_counter()
                # Per-epoch permutations with the remainder dropped so all minibatch
                # shapes are static. Adaptive mode uses the same chunk_epochs every
                # time, so XLA can reuse the compiled scanned training graph.
                epoch_indices = range(epochs_run + 1, epochs_run + int(chunk_epochs) + 1)
                perm_rows = np.stack(
                    [
                        (self.audit_permutation(epoch_idx, n_samples) if hasattr(self, "audit_permutation") else fit_np_rng("epoch", epoch_idx, "permutation").permutation(n_samples))[: n_full_batches * batch_size]
                        .reshape(
                            n_full_batches, batch_size
                        )
                        for epoch_idx in epoch_indices
                    ]
                ).astype(np.int32)
                key_train = fit_key("scan_chunk", chunk_idx, "train")
                if hasattr(self, "audit_before_fit"):
                    self.audit_before_fit(epochs_run + 1, params, perm_rows, key_train)
                params, opt_state, epoch_losses = train_all(
                    params, opt_state, x, y, w, jnp.asarray(perm_rows), key_train, status_1q, status_2q
                )
                if hasattr(self, "audit_after_fit"):
                    self.audit_after_fit(epochs_run + 1, params, epoch_losses)
                chunk_train_evaluations = int(chunk_epochs) * n_full_batches * batch_size
                train_model_evaluations += chunk_train_evaluations
                optimizer_updates += int(chunk_epochs) * n_full_batches
                self._unshifted_loss_evaluations += chunk_train_evaluations
                expected_unshifted_loss_evaluations += chunk_train_evaluations

                should_stop = False
                for loss_value in np.asarray(epoch_losses, dtype=np.float64).tolist():
                    epochs_run += 1
                    epoch_loss_history.append(float(loss_value))
                    final_epoch_loss_before_restoration = float(loss_value)
                    decision = _record_completed_epoch(
                        epoch_loss=float(loss_value),
                        current_params=params,
                        current_opt_state=opt_state,
                        epoch_rng_state=key_train,
                        at_max_epochs=epochs_run >= int(self.n_epochs),
                    )
                    should_stop = bool(decision.should_stop or target_confirmed)

                last_loss = None if not epoch_loss_history else float(epoch_loss_history[-1])
                if self._should_log_epoch(epochs_run, converged=should_stop):
                    self._log(
                        f"chunk {chunk_idx}/{len(chunk_sizes)} done "
                        f"epochs_run={epochs_run}/{int(self.n_epochs)} "
                        f"loss={last_loss} stop={should_stop} "
                        f"elapsed={time.perf_counter() - chunk_t0:.2f}s"
                    )

                if should_stop:
                    break
        else:
            if self._uses_parameter_shift_gradient():
                optimizer, train_step = _cached_parameter_shift_optimizer_and_train_step(
                    learning_rate=float(self.learning_rate),
                    grad_clip_norm=float(self.grad_clip_norm),
                    sigma=float(self.sigma),
                    parameter_shift_chunk_size=self.parameter_shift_chunk_size,
                    **self._predict_config(),
                )
                shifted_evaluations_per_example = 2 * int(self._shifted_gate_occurrence_count())
            else:
                optimizer, train_step = _cached_optimizer_and_train_step(
                    learning_rate=float(self.learning_rate),
                    grad_clip_norm=float(self.grad_clip_norm),
                    sigma=float(self.sigma),
                    **self._predict_config(),
                )
                shifted_evaluations_per_example = 0
            initial_opt_state = optimizer.init(params)
            opt_state = _restore_optimizer_state(initial_opt_state)
            best_opt_state = _restore_best_optimizer_state(initial_opt_state)

            train_model_evaluations = int(
                0
                if resume_state is None
                else resume_state["resource_ledger"]["training_model_evaluations"]
            )
            for epoch_idx in range(epochs_run + 1, int(self.n_epochs) + 1):
                epoch_t0 = time.perf_counter()
                perm = fit_np_rng("epoch", epoch_idx, "permutation").permutation(n_samples)
                epoch_losses = []
                for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
                    idx = perm[start : start + batch_size]
                    if idx.size == 0:
                        continue

                    idx_jax = jnp.asarray(idx)
                    xb = x[idx_jax]
                    unshifted_batch_evaluations = int(idx.size)
                    shifted_batch_evaluations = int(idx.size) * int(shifted_evaluations_per_example)
                    train_model_evaluations += (
                        unshifted_batch_evaluations + shifted_batch_evaluations
                    )
                    optimizer_updates += 1
                    self._unshifted_loss_evaluations += unshifted_batch_evaluations
                    self._shifted_gradient_evaluations += shifted_batch_evaluations
                    expected_unshifted_loss_evaluations += unshifted_batch_evaluations

                    yb = y[idx_jax]
                    wb = w[idx_jax]

                    key_batch = fit_key("epoch", epoch_idx, "batch", batch_idx, "train")
                    params, opt_state, batch_loss = train_step(
                        params, opt_state, xb, yb, wb, key_batch, status_1q, status_2q
                    )
                    epoch_losses.append(batch_loss)

                if epoch_losses:
                    epochs_run += 1
                    epoch_loss = float(jnp.mean(jnp.stack(epoch_losses)))
                    epoch_loss_history.append(epoch_loss)
                    final_epoch_loss_before_restoration = epoch_loss
                    decision = _record_completed_epoch(
                        epoch_loss=epoch_loss,
                        current_params=params,
                        current_opt_state=opt_state,
                        epoch_rng_state=key_batch,
                        at_max_epochs=epochs_run >= int(self.n_epochs),
                    )
                    should_stop = bool(decision.should_stop or target_confirmed)
                    if self._should_log_epoch(epochs_run, converged=should_stop):
                        self._log(
                            f"epoch {epochs_run}/{int(self.n_epochs)} "
                            f"loss={epoch_loss:.8g} stop={should_stop} "
                            f"elapsed={time.perf_counter() - epoch_t0:.2f}s"
                        )

                if should_stop:
                    break

        current_params_before_restoration = jnp.array(params)
        current_opt_state_before_restoration = jax.tree_util.tree_map(
            lambda value: jnp.array(value),
            opt_state,
        )
        best_parameters_restored = bool(
            self.restore_best_parameters
            and not target_confirmed
            and best_params is not None
        )
        if best_parameters_restored:
            params = jnp.array(best_params)
            opt_state = jax.tree_util.tree_map(
                lambda value: jnp.array(value),
                best_opt_state,
            )
        self.params_ = params

        if int(self._unshifted_loss_evaluations) != int(expected_unshifted_loss_evaluations):
            raise AssertionError(
                "unshifted_loss_evaluations must count individual "
                "example-expectation evaluations: "
                f"observed={int(self._unshifted_loss_evaluations)}, "
                f"expected_sum_batches={int(expected_unshifted_loss_evaluations)}."
            )

        if self.compute_fit_diagnostic:
            diagnostic_t0 = time.perf_counter()
            self._log("fit diagnostic start")
            train_model_evaluations += int(n_samples)
            self._training_monitor_evaluations += int(n_samples)
            key_eval = fit_key("fit_diagnostic")
            full_loss_fn = _cached_loss_fn(sigma=float(self.sigma), **self._predict_config())
            if hasattr(self, "audit_before_fit_diagnostic"):
                self.audit_before_fit_diagnostic(n_samples, key_eval)
            full_loss = full_loss_fn(
                self.params_, x, y, w, key_eval, status_1q=status_1q, status_2q=status_2q
            )
            weighted_surrogate_loss_t = float(full_loss)
            if hasattr(self, "audit_after_fit_diagnostic"):
                self.audit_after_fit_diagnostic(weighted_surrogate_loss_t)
            self._log(
                f"fit diagnostic complete loss={weighted_surrogate_loss_t:.8g} "
                f"elapsed={time.perf_counter() - diagnostic_t0:.2f}s"
            )
        else:
            weighted_surrogate_loss_t = float("nan")

        self._train_model_evaluations = int(train_model_evaluations)

        convergence_summary = convergence_tracker.state_dict()
        convergence_summary.update(
            {
                "max_epochs": int(self.n_epochs),
                "fixed_min_epochs": None,
                "epochs_run": int(epochs_run),
                "monitored_loss": str(self.monitored_loss),
                "restore_best_parameters": bool(self.restore_best_parameters),
                "best_parameters_restored": bool(best_parameters_restored),
                "final_epoch_loss_before_restoration": (
                    None
                    if final_epoch_loss_before_restoration is None
                    else float(final_epoch_loss_before_restoration)
                ),
                "restored_best_epoch_loss": (
                    convergence_tracker.best_epoch_loss
                    if best_parameters_restored
                    else None
                ),
                "stop_reason": str(training_stop_reason),
                "scan_training": bool(use_scan),
                "optimizer_updates": int(optimizer_updates),
                "target_check_frequency_epochs": int(
                    self.target_check_frequency_epochs
                ),
                "target_check_history": list(target_check_history),
                "target_confirmed": bool(target_confirmed),
                "target_confirmation": (
                    None if target_confirmation is None else dict(target_confirmation)
                ),
                "resume_state": {
                    "training_stopping_definition": (
                        BEST_LOSS_NO_IMPROVEMENT_DEFINITION
                    ),
                    "fit_seed": int(seed),
                    "current_epoch": int(epochs_run),
                    "current_parameters": np.asarray(
                        current_params_before_restoration
                    ).tolist(),
                    "best_parameters": (
                        None
                        if best_params is None
                        else np.asarray(best_params).tolist()
                    ),
                    "current_optimizer_state_leaves": [
                        np.asarray(value).tolist()
                        for value in jax.tree_util.tree_leaves(
                            current_opt_state_before_restoration
                        )
                    ],
                    "best_optimizer_state_leaves": (
                        None
                        if best_opt_state is None
                        else [
                            np.asarray(value).tolist()
                            for value in jax.tree_util.tree_leaves(best_opt_state)
                        ]
                    ),
                    "best_epoch_loss": convergence_tracker.best_epoch_loss,
                    "best_epoch": int(convergence_tracker.best_epoch),
                    "epochs_without_improvement": int(
                        convergence_tracker.epochs_without_improvement
                    ),
                    "rng_state": (
                        None
                        if current_rng_state is None
                        else np.asarray(current_rng_state).tolist()
                    ),
                    "best_rng_state": (
                        None
                        if best_rng_state is None
                        else np.asarray(best_rng_state).tolist()
                    ),
                    "resource_ledger": {
                        "epochs_run": int(epochs_run),
                        "optimizer_updates": int(optimizer_updates),
                        "training_model_evaluations": int(
                            train_model_evaluations
                        ),
                        "gradient_evaluations": int(optimizer_updates),
                        "shifted_gradient_model_evaluations": int(
                            self._shifted_gradient_evaluations
                        ),
                        "unshifted_loss_evaluations": int(
                            self._unshifted_loss_evaluations
                        ),
                        "training_monitor_evaluations": int(
                            self._training_monitor_evaluations
                        ),
                    },
                    "stopping_state": convergence_tracker.state_dict(),
                },
            }
        )
        self.training_resume_state_ = dict(convergence_summary["resume_state"])
        self.last_training_stopping_ = dict(convergence_summary)
        self.optimizer_updates_ = int(optimizer_updates)

        self._log(
            f"fit complete epochs_run={epochs_run}/{int(self.n_epochs)} "
            f"stop_reason={training_stop_reason} "
            f"train_evals={int(train_model_evaluations)} "
            f"elapsed={time.perf_counter() - fit_t0:.2f}s"
        )

        return WeakLearnerFitResult(
            weighted_surrogate_loss_t=weighted_surrogate_loss_t,
            epochs_run=int(epochs_run),
            converged=bool(
                training_stop_reason.startswith("no_improvement_patience_")
                or target_confirmed
            ),
            convergence_diagnostics=convergence_summary,
            target_confirmed=bool(target_confirmed),
            target_confirmation=target_confirmation,
        )

    def predict_scores_jax(
        self,
        x: Array,
        rng: Optional[np.random.Generator] = None,
    ) -> jax.Array:
        """In-graph batch prediction returning a device ``jax.Array`` (Opt 1).

        Unlike :meth:`predict_score`, this does not force a device->host transfer.
        The boosting loop uses it so per-round train/val/test scoring stays on the
        accelerator and feeds directly into the fused round kernel, eliminating the
        host round-trips that ``predict_score`` incurs.
        """
        if self.params_ is None:
            raise RuntimeError("Model is not fitted. Call fit_weighted first.")

        x_j = jnp.asarray(x, dtype=_DTYPE)
        if x_j.ndim != 2:
            raise ValueError("x must be rank-2 with shape [n_samples, n_features].")

        if self._prediction_phase == "train":
            self._train_model_evaluations += int(x_j.shape[0])
        else:
            self._inference_model_evaluations += int(x_j.shape[0])

        if rng is None:
            rng = np.random.default_rng(0)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        key = jax.random.PRNGKey(seed)

        status_1q, status_2q = self._mc_status_args()
        predict = _cached_predict_batch(**self._predict_config())
        return predict(self.params_, x_j, key, status_1q=status_1q, status_2q=status_2q)

    def predict_deterministic_mean_scores_jax(self, x: Array) -> jax.Array:
        """Return the conditional mean score without finite-shot sampling.

        This is an uncharged stopping diagnostic.  For the density-matrix Stage-C
        protocol it is the exact expectation under the configured physical noise
        channel, not a noiseless circuit score and not a stored finite-shot point
        estimate.  Monte-Carlo trajectory backends are deliberately rejected
        because a single fixed trajectory table is not the unconditional mean.
        """
        if self.params_ is None:
            raise RuntimeError("Model is not fitted. Call fit_weighted first.")
        if self.noise_backend in {"mc_vmap", "mc_streaming"}:
            raise ValueError(
                "deterministic-mean prediction is not defined for Monte-Carlo "
                "trajectory backends"
            )
        x_j = jnp.asarray(x, dtype=_DTYPE)
        if x_j.ndim != 2:
            raise ValueError("x must be rank-2 with shape [n_samples, n_features].")
        config = self._predict_config()
        config["n_shots_eff"] = 0
        predict = _cached_predict_batch(**config)
        key = jax.random.PRNGKey(0)
        return predict(self.params_, x_j, key, status_1q=None, status_2q=None)

    def predict_deterministic_mean_score(self, x: Array) -> Array:
        """Host wrapper for :meth:`predict_deterministic_mean_scores_jax`."""
        return np.asarray(self.predict_deterministic_mean_scores_jax(x), dtype=np.float64)

    def predict_score(
        self,
        x: Array,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        # Host-facing wrapper: materialize the in-graph scores to NumPy.
        return np.asarray(self.predict_scores_jax(x, rng=rng), dtype=np.float64)

    def predict_label(
        self,
        x: Array,
        threshold: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        score = self.predict_score(x, rng=rng)
        return np.where(score >= threshold, 1, -1).astype(np.int8)

    def state_dict(self) -> Mapping[str, Any]:
        if self.params_ is None:
            raise RuntimeError("Cannot serialize an unfitted weak learner.")
        return {
            "weak_learner_type": "weighted_pqc_binary",
            "n_qubits": int(self.n_qubits),
            "n_layers": int(self.n_layers),
            "n_shots_eff": int(self.n_shots_eff),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "grad_clip_norm": float(self.grad_clip_norm),
            "sigma": float(self.sigma),
            "scan_training": bool(self.scan_training),
            "adaptive_epoch_stopping": bool(self.adaptive_epoch_stopping),
            "adaptive_epoch_window_size": int(self.adaptive_epoch_window_size),
            "adaptive_epoch_min_epochs": int(self.adaptive_epoch_min_epochs),
            "adaptive_epoch_patience": int(self.adaptive_epoch_patience),
            "adaptive_epoch_scale": float(self.adaptive_epoch_scale),
            "adaptive_epoch_chunk_size": int(self.adaptive_epoch_chunk_size),
            "training_stopping_definition": str(self.training_stopping_definition),
            "fixed_min_epochs": self.fixed_min_epochs,
            "early_stopping_enabled": bool(self.early_stopping_enabled),
            "early_stopping_patience_epochs": int(
                self.early_stopping_patience_epochs
            ),
            "early_stopping_check_frequency_epochs": int(
                self.early_stopping_check_frequency_epochs
            ),
            "early_stopping_min_delta": float(self.early_stopping_min_delta),
            "restore_best_parameters": bool(self.restore_best_parameters),
            "optimizer": "optax.adam",
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.999,
            "optimizer_epsilon": 1e-8,
            "optimizer_eps_root": 0.0,
            "updates_per_epoch": (
                None
                if self.updates_per_epoch is None
                else int(self.updates_per_epoch)
            ),
            "gradient_clipping": "global_norm",
            "monitored_loss": str(self.monitored_loss),
            "intra_round_target_stopping": bool(self.intra_round_target_stopping),
            "target_check_frequency_epochs": int(
                self.target_check_frequency_epochs
            ),
            "check_on_training_stop": bool(self.check_on_training_stop),
            "optimizer_updates": int(self.optimizer_updates_),
            "gradient_evaluations": int(self.optimizer_updates_),
            "training_resume_state": (
                None
                if self.training_resume_state_ is None
                else dict(self.training_resume_state_)
            ),
            "training_stopping": (
                None
                if self.last_training_stopping_ is None
                else dict(self.last_training_stopping_)
            ),
            "training_accuracy_target": (
                None if self.training_accuracy_target is None else float(self.training_accuracy_target)
            ),
            "ansatz_kind": str(self.ansatz_kind),
            "n_reuploads": int(self.n_reuploads),
            "paper_data_chunk_mode": str(self.paper_data_chunk_mode),
            "paper_observable": str(self.paper_observable),
            "noise_backend": str(self.noise_backend),
            "n_trajectories": int(self.n_trajectories),
            "trajectory_batch_size": int(self.trajectory_batch_size),
            "noise_strength_1q": float(self.noise_strength_1q),
            "noise_strength_2q": float(self.noise_strength_2q),
            "mc_apply_two_qubit_noise": bool(self.mc_apply_two_qubit_noise),
            "mc_status_mode": str(self.mc_status_mode),
            "measurement_sampling_distribution": str(self.measurement_sampling_distribution),
            "measurement_gradient_mode": str(self.measurement_gradient_mode),
            "parameter_shift_chunk_size": self.parameter_shift_chunk_size,
            "compute_fit_diagnostic": bool(self.compute_fit_diagnostic),
            "mc_status_1q": (
                None if self._mc_status_1q_ is None else np.asarray(self._mc_status_1q_, dtype=np.float32).tolist()
            ),
            "mc_status_2q": (
                None if self._mc_status_2q_ is None else np.asarray(self._mc_status_2q_, dtype=np.float32).tolist()
            ),
            "raw_shot_overhead": float(self.raw_shot_overhead),
            "verbose": bool(self.verbose),
            "log_every_epochs": int(self.log_every_epochs),
            "param_init_scheme": str(self.param_init_scheme),
            "param_init_std": float(self.param_init_std),
            "param_init_encoding_mean": float(self.param_init_encoding_mean),
            "param_init_encoding_std": float(self.param_init_encoding_std),
            "param_init_variational_mean": float(self.param_init_variational_mean),
            "param_init_variational_std": float(self.param_init_variational_std),
            "param_init_final_mean": float(self.param_init_final_mean),
            "param_init_final_std": float(self.param_init_final_std),
            "param_init_teacher_mean": (
                None
                if self.param_init_teacher_mean is None
                else np.asarray(self.param_init_teacher_mean, dtype=np.float32).tolist()
            ),
            "param_init_teacher_std": float(self.param_init_teacher_std),
            "param_init_clip_abs": (
                None if self.param_init_clip_abs is None else float(self.param_init_clip_abs)
            ),
            "params": np.asarray(self.params_, dtype=np.float32).tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "WeightedPQCBinaryWeakLearner":
        learner = cls(
            n_qubits=int(state["n_qubits"]),
            n_layers=int(state["n_layers"]),
            n_shots_eff=int(state.get("n_shots_eff", int(1e4))),
            n_epochs=int(state.get("n_epochs", 500)),
            batch_size=int(state.get("batch_size", 64)),
            learning_rate=float(state.get("learning_rate", 1e-2)),
            grad_clip_norm=float(state.get("grad_clip_norm", 1.0)),
            sigma=float(state.get("sigma", 0.5)),
            scan_training=bool(state.get("scan_training", False)),
            adaptive_epoch_stopping=bool(state.get("adaptive_epoch_stopping", False)),
            adaptive_epoch_window_size=int(state.get("adaptive_epoch_window_size", 50)),
            adaptive_epoch_min_epochs=int(state.get("adaptive_epoch_min_epochs", 100)),
            adaptive_epoch_patience=int(state.get("adaptive_epoch_patience", 2)),
            adaptive_epoch_scale=float(state.get("adaptive_epoch_scale", 0.75)),
            adaptive_epoch_chunk_size=int(state.get("adaptive_epoch_chunk_size", 10)),
            training_stopping_definition=str(
                state.get(
                    "training_stopping_definition",
                    BEST_LOSS_NO_IMPROVEMENT_DEFINITION,
                )
            ),
            fixed_min_epochs=state.get("fixed_min_epochs"),
            early_stopping_enabled=bool(state.get("early_stopping_enabled", True)),
            early_stopping_patience_epochs=int(
                state.get("early_stopping_patience_epochs", 20)
            ),
            early_stopping_check_frequency_epochs=int(
                state.get("early_stopping_check_frequency_epochs", 1)
            ),
            early_stopping_min_delta=float(
                state.get("early_stopping_min_delta", 0.0)
            ),
            restore_best_parameters=bool(
                state.get("restore_best_parameters", True)
            ),
            updates_per_epoch=(
                None
                if state.get("updates_per_epoch") is None
                else int(state["updates_per_epoch"])
            ),
            monitored_loss=str(
                state.get(
                    "monitored_loss",
                    "adaboost_weighted_epoch_average_smooth_hinge_PsiG_sigma_0p5",
                )
            ),
            intra_round_target_stopping=bool(
                state.get("intra_round_target_stopping", True)
            ),
            target_check_frequency_epochs=int(
                state.get("target_check_frequency_epochs", 10)
            ),
            check_on_training_stop=bool(state.get("check_on_training_stop", True)),
            training_accuracy_target=(
                None
                if state.get("training_accuracy_target") is None
                else float(state["training_accuracy_target"])
            ),
            ansatz_kind=str(state.get("ansatz_kind", "legacy_reuploading")),
            n_reuploads=int(state.get("n_reuploads", 1)),
            paper_data_chunk_mode=str(state.get("paper_data_chunk_mode", "cyclic")),
            paper_observable=str(state.get("paper_observable", "z0")),
            noise_backend=str(state.get("noise_backend", "effective_shot")),
            n_trajectories=int(state.get("n_trajectories", 1)),
            trajectory_batch_size=int(state.get("trajectory_batch_size", 16)),
            noise_strength_1q=float(state.get("noise_strength_1q", 0.0)),
            noise_strength_2q=float(state.get("noise_strength_2q", 0.0)),
            mc_apply_two_qubit_noise=bool(state.get("mc_apply_two_qubit_noise", True)),
            mc_status_mode=str(state.get("mc_status_mode", "fresh")),
            measurement_sampling_distribution=str(state.get("measurement_sampling_distribution", "binomial")),
            measurement_gradient_mode=str(
                state.get("measurement_gradient_mode", "straight_through_expectation")
            ),
            parameter_shift_chunk_size=state.get("parameter_shift_chunk_size", 0),
            compute_fit_diagnostic=bool(state.get("compute_fit_diagnostic", True)),
            raw_shot_overhead=float(state.get("raw_shot_overhead", 1.0)),
            verbose=bool(state.get("verbose", False)),
            log_every_epochs=int(state.get("log_every_epochs", 50)),
            param_init_scheme=str(state.get("param_init_scheme", "block_normal")),
            param_init_std=float(state.get("param_init_std", 1.0)),
            param_init_encoding_mean=float(state.get("param_init_encoding_mean", 1.0)),
            param_init_encoding_std=float(state.get("param_init_encoding_std", 0.1)),
            param_init_variational_mean=float(state.get("param_init_variational_mean", 0.0)),
            param_init_variational_std=float(state.get("param_init_variational_std", 0.1)),
            param_init_final_mean=float(state.get("param_init_final_mean", 0.0)),
            param_init_final_std=float(state.get("param_init_final_std", 0.1)),
            param_init_teacher_mean=(
                None
                if state.get("param_init_teacher_mean") is None
                else np.asarray(state.get("param_init_teacher_mean"), dtype=np.float32)
            ),
            param_init_teacher_std=float(state.get("param_init_teacher_std", 0.1)),
            param_init_clip_abs=(
                None
                if state.get("param_init_clip_abs") is None
                else float(state.get("param_init_clip_abs"))
            ),
        )
        learner.params_ = jnp.asarray(state["params"], dtype=_DTYPE)
        learner.optimizer_updates_ = int(state.get("optimizer_updates", 0))
        learner.training_resume_state_ = state.get("training_resume_state")
        learner.last_training_stopping_ = state.get("training_stopping")
        if state.get("mc_status_1q") is not None and state.get("mc_status_2q") is not None:
            learner._mc_status_1q_ = jnp.asarray(state["mc_status_1q"], dtype=_DTYPE)
            learner._mc_status_2q_ = jnp.asarray(state["mc_status_2q"], dtype=_DTYPE)
        return learner
