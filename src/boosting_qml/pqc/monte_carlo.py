"""Monte Carlo trajectory backends for noisy legacy reuploading PQCs.

The functions here intentionally keep two equivalent execution styles:

- ``*_vmap_*`` mirrors the original prototype idea and vectorizes all
  trajectories at once.
- ``*_streaming_*`` evaluates trajectories in small blocks and accumulates the
  scalar expectation. This is the A100-friendly path because it avoids keeping
  all trajectory circuit activations live during autodiff.

Both paths consume the same pre-sampled TensorCircuit ``status`` tensors, so
unit tests can compare them exactly up to floating-point differences.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy_compat import patch_numpy_for_tensorcircuit

patch_numpy_for_tensorcircuit()

import tensorcircuit as tc

from boosting_qml.pqc.measurement import sample_single_pauli_outcomes

K = tc.set_backend("jax")

Array = jax.Array
_DTYPE = jnp.float32


def _entangler_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Ring-CZ connectivity used by the legacy teacher-student ansatz."""

    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    if n_qubits == 1:
        return ()

    pairs = [(q, q + 1) for q in range(n_qubits - 1)]
    if n_qubits != 2:
        pairs.append((n_qubits - 1, 0))
    return tuple(pairs)


def legacy_reuploading_noisy_gate_counts(n_qubits: int, n_layers: int) -> dict[str, int]:
    """Return noisy-location counts used by the legacy reuploading simulator."""

    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}.")
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")
    return {
        "G1": int((4 * n_layers + 3) * n_qubits),
        "G2": int(n_layers * len(_entangler_pairs(n_qubits))),
    }

_TWO_QUBIT_NON_ID_PAULIS: tuple[tuple[int, int], ...] = tuple(
    (pa, pb) for pa in range(4) for pb in range(4) if not (pa == 0 and pb == 0)
)
_TWO_QUBIT_PAULIS: tuple[tuple[int, int], ...] = ((0, 0),) + _TWO_QUBIT_NON_ID_PAULIS


def _apply_one_qubit_pauli_state(state: Array, qubit: int, pauli: int, n_qubits: int) -> Array:
    """Apply I/X/Y/Z to a flat statevector using JAX array operations."""

    if pauli == 0:
        return state

    tensor = jnp.reshape(state, (2,) * int(n_qubits))
    tensor = jnp.moveaxis(tensor, int(qubit), 0)

    if pauli == 1:  # X
        updated = jnp.flip(tensor, axis=0)
    elif pauli == 2:  # Y = [[0, -i], [i, 0]]
        imag = jnp.asarray(1j, dtype=tensor.dtype)
        updated = jnp.stack((-imag * tensor[1], imag * tensor[0]), axis=0)
    elif pauli == 3:  # Z
        z = jnp.asarray([1.0, -1.0], dtype=tensor.dtype).reshape(
            (2,) + (1,) * (int(n_qubits) - 1)
        )
        updated = tensor * z
    else:  # pragma: no cover - only closed-over constants 0..3 are used.
        raise ValueError(f"Unsupported Pauli index {pauli}.")

    updated = jnp.moveaxis(updated, 0, int(qubit))
    return jnp.reshape(updated, (-1,))


def _apply_two_qubit_depolarizing_status_state(
    state: Array,
    qa: int,
    qb: int,
    status: Array,
    *,
    n_qubits: int,
    p_each: float,
) -> Array:
    """Sample the 2q depolarizing Pauli using ``status`` and apply it to state.

    The project convention is D_epsilon(rho) = (1-epsilon) rho + epsilon I/d.
    For two qubits this is implemented by applying each non-identity two-qubit
    Pauli string with probability epsilon / 16, and identity otherwise.
    """

    if float(p_each) <= 0.0:
        return state

    p_each_arr = jnp.asarray(float(p_each), dtype=jnp.asarray(status).dtype)
    non_id_mass = 15.0 * p_each_arr
    non_id_idx = jnp.floor(jnp.asarray(status) / p_each_arr).astype(jnp.int32)
    pauli_code = jnp.where(status < non_id_mass, 1 + non_id_idx, 0)
    pauli_code = jnp.clip(pauli_code, 0, 15)

    branches = []
    for pa, pb in _TWO_QUBIT_PAULIS:
        def _branch(s, pa=pa, pb=pb):
            s = _apply_one_qubit_pauli_state(s, int(qa), int(pa), int(n_qubits))
            return _apply_one_qubit_pauli_state(s, int(qb), int(pb), int(n_qubits))

        branches.append(_branch)

    return jax.lax.switch(pauli_code, branches, state)


def reuploading_one_qubit_status_shape(n_qubits: int, n_layers: int) -> tuple[int, int]:
    """Status shape for all one-qubit noisy gates in one trajectory."""

    return (4 * int(n_layers) + 3, int(n_qubits))


def reuploading_two_qubit_status_shape(n_qubits: int, n_layers: int) -> tuple[int, int]:
    """Status shape for exact two-qubit depolarizing noise in one trajectory."""

    return (int(n_layers), len(_entangler_pairs(int(n_qubits))))


def make_reuploading_mc_statuses(
    key: Array,
    *,
    n_trajectories: int,
    n_qubits: int,
    n_layers: int,
    dtype: jnp.dtype = _DTYPE,
) -> tuple[Array, Array]:
    """Sample stochastic channel statuses for legacy reuploading trajectories."""

    n_trajectories = int(n_trajectories)
    if n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be positive, got {n_trajectories}.")

    key_1q, key_2q = jax.random.split(key)
    status_1q = jax.random.uniform(
        key_1q,
        shape=(n_trajectories,) + reuploading_one_qubit_status_shape(n_qubits, n_layers),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    status_2q = jax.random.uniform(
        key_2q,
        shape=(n_trajectories,) + reuploading_two_qubit_status_shape(n_qubits, n_layers),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    return status_1q, status_2q


def _validate_legacy_inputs(
    x: Array,
    params: Array,
    status_1q: Array,
    status_2q: Array,
    *,
    n_qubits: int,
    n_layers: int,
) -> None:
    expected_params = (4 * int(n_layers) + 3, int(n_qubits))
    expected_1q = reuploading_one_qubit_status_shape(n_qubits, n_layers)
    expected_2q = reuploading_two_qubit_status_shape(n_qubits, n_layers)

    if tuple(params.shape) != expected_params:
        raise ValueError(f"Expected params shape {expected_params}, got {tuple(params.shape)}.")
    if int(x.reshape(-1).shape[0]) != int(n_qubits):
        raise ValueError(f"Expected x length {int(n_qubits)}, got {int(x.reshape(-1).shape[0])}.")
    if tuple(status_1q.shape) != expected_1q:
        raise ValueError(f"Expected status_1q shape {expected_1q}, got {tuple(status_1q.shape)}.")
    if tuple(status_2q.shape) != expected_2q:
        raise ValueError(f"Expected status_2q shape {expected_2q}, got {tuple(status_2q.shape)}.")


def noisy_reuploading_score_one_status(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Evaluate one pure-state depolarizing trajectory.

    The one-qubit channel is sampled after every one-qubit gate. The two-qubit
    channel samples one exact two-qubit Pauli string after each entangler using
    a JIT-safe statevector update rather than TensorCircuit's non-JIT-safe
    ``generaldepolarizing(..., num_qubits=2)`` helper.
    """

    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    x = jnp.asarray(x, dtype=_DTYPE).reshape(-1)
    params = jnp.asarray(params, dtype=_DTYPE)
    status_1q = jnp.asarray(status_1q, dtype=_DTYPE)
    status_2q = jnp.asarray(status_2q, dtype=_DTYPE)
    _validate_legacy_inputs(x, params, status_1q, status_2q, n_qubits=n_qubits, n_layers=n_layers)

    px1 = py1 = pz1 = float(noise_strength_1q) / 4.0
    # The explicit two-qubit Pauli sampler uses p as the probability of each
    # non-identity two-qubit Pauli string. Our project-level
    # convention uses D_epsilon(rho) = (1 - epsilon) rho + epsilon I / d, so
    # each of the 15 non-identity two-qubit Paulis has probability epsilon / 16.
    p2_each = float(noise_strength_2q) / 16.0
    pairs = _entangler_pairs(n_qubits)

    layer_params = jnp.stack(
        [
            params[0 : 4 * n_layers : 4, :],
            params[1 : 4 * n_layers : 4, :],
            params[2 : 4 * n_layers : 4, :],
            params[3 : 4 * n_layers : 4, :],
        ],
        axis=1,
    )
    layer_status_1q = status_1q[: 4 * n_layers, :].reshape(n_layers, 4, n_qubits)
    final_status_1q = status_1q[4 * n_layers : 4 * n_layers + 3, :]

    s0 = tc.Circuit(n_qubits).state()

    def layer_loop(s_in, layer_data):
        layer_param_block, st1_block, st2_block = layer_data
        lambda_l = layer_param_block[0]
        rx_l = layer_param_block[1]
        ry_l = layer_param_block[2]
        rz_l = layer_param_block[3]
        c_ = tc.Circuit(n_qubits, inputs=s_in)

        for q in range(n_qubits):
            c_.rx(q, theta=rx_l[q])
            if noise_strength_1q > 0.0:
                c_.depolarizing(q, px=px1, py=py1, pz=pz1, status=st1_block[0, q])
            c_.ry(q, theta=ry_l[q])
            if noise_strength_1q > 0.0:
                c_.depolarizing(q, px=px1, py=py1, pz=pz1, status=st1_block[1, q])
            c_.rz(q, theta=rz_l[q])
            if noise_strength_1q > 0.0:
                c_.depolarizing(q, px=px1, py=py1, pz=pz1, status=st1_block[2, q])

        for pair_idx, (qa, qb) in enumerate(pairs):
            c_.cz(qa, qb)
            if apply_two_qubit_noise and noise_strength_2q > 0.0:
                noisy_state = _apply_two_qubit_depolarizing_status_state(
                    c_.state(),
                    qa,
                    qb,
                    st2_block[pair_idx],
                    n_qubits=n_qubits,
                    p_each=p2_each,
                )
                c_ = tc.Circuit(n_qubits, inputs=noisy_state)

        for q in range(n_qubits):
            c_.rx(q, theta=x[q] * lambda_l[q])
            if noise_strength_1q > 0.0:
                c_.depolarizing(q, px=px1, py=py1, pz=pz1, status=st1_block[3, q])

        return c_.state()

    s_final = K.scan(layer_loop, (layer_params, layer_status_1q, status_2q), s0)

    c_final = tc.Circuit(n_qubits, inputs=s_final)
    final_offset = 4 * n_layers
    for q in range(n_qubits):
        c_final.rx(q, theta=params[final_offset + 0, q])
        if noise_strength_1q > 0.0:
            c_final.depolarizing(q, px=px1, py=py1, pz=pz1, status=final_status_1q[0, q])
        c_final.ry(q, theta=params[final_offset + 1, q])
        if noise_strength_1q > 0.0:
            c_final.depolarizing(q, px=px1, py=py1, pz=pz1, status=final_status_1q[1, q])
        c_final.rz(q, theta=params[final_offset + 2, q])
        if noise_strength_1q > 0.0:
            c_final.depolarizing(q, px=px1, py=py1, pz=pz1, status=final_status_1q[2, q])

    if observable == "global_z":
        score = c_final.expectation_ps(z=np.arange(n_qubits))
    elif observable == "z0":
        score = c_final.expectation_ps(z=[0])
    else:
        raise ValueError(f"Unsupported observable '{observable}'. Expected 'global_z' or 'z0'.")
    return jnp.real(score)


def noisy_reuploading_score_vmap_status(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Original all-at-once trajectory vectorization backend."""

    scores = noisy_reuploading_trajectory_scores_status(
        x,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
        status_1q=status_1q,
        status_2q=status_2q,
        noise_strength_1q=noise_strength_1q,
        noise_strength_2q=noise_strength_2q,
        apply_two_qubit_noise=apply_two_qubit_noise,
        observable=observable,
    )
    return jnp.mean(scores)


def noisy_reuploading_trajectory_scores_status(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Return one noisy expectation for each supplied channel trajectory."""

    return jax.vmap(
        lambda st1, st2: noisy_reuploading_score_one_status(
            x,
            n_qubits=n_qubits,
            params=params,
            n_layers=n_layers,
            status_1q=st1,
            status_2q=st2,
            noise_strength_1q=noise_strength_1q,
            noise_strength_2q=noise_strength_2q,
            apply_two_qubit_noise=apply_two_qubit_noise,
            observable=observable,
        )
    )(status_1q, status_2q)


def noisy_reuploading_sampled_score_vmap_status(
    key_measurement: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
    measurement_gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Average one Pauli measurement outcome per supplied trajectory."""

    scores = noisy_reuploading_trajectory_scores_status(
        x,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
        status_1q=status_1q,
        status_2q=status_2q,
        noise_strength_1q=noise_strength_1q,
        noise_strength_2q=noise_strength_2q,
        apply_two_qubit_noise=apply_two_qubit_noise,
        observable=observable,
    )
    outcomes = sample_single_pauli_outcomes(
        scores,
        key_measurement,
        gradient_mode=measurement_gradient_mode,
    )
    return jnp.mean(outcomes)


def noisy_reuploading_score_streaming_status(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    trajectory_batch_size: int = 16,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Streaming trajectory backend with small-vectorized trajectory blocks."""

    n_trajectories = int(status_1q.shape[0])
    trajectory_batch_size = int(trajectory_batch_size)
    if n_trajectories <= 0:
        raise ValueError("status_1q must contain at least one trajectory.")
    if trajectory_batch_size <= 0:
        raise ValueError("trajectory_batch_size must be positive.")
    if int(status_2q.shape[0]) != n_trajectories:
        raise ValueError("status_1q and status_2q must have the same trajectory count.")

    num_batches = (n_trajectories + trajectory_batch_size - 1) // trajectory_batch_size
    padded_n = num_batches * trajectory_batch_size
    pad_n = padded_n - n_trajectories
    status_1q_padded = jnp.pad(status_1q, ((0, pad_n), (0, 0), (0, 0)))
    status_2q_padded = jnp.pad(status_2q, ((0, pad_n), (0, 0), (0, 0)))

    def body(batch_idx, acc):
        start = batch_idx * trajectory_batch_size
        st1_batch = jax.lax.dynamic_slice_in_dim(
            status_1q_padded,
            start,
            trajectory_batch_size,
            axis=0,
        )
        st2_batch = jax.lax.dynamic_slice_in_dim(
            status_2q_padded,
            start,
            trajectory_batch_size,
            axis=0,
        )
        scores = jax.vmap(
            lambda st1, st2: noisy_reuploading_score_one_status(
                x,
                n_qubits=n_qubits,
                params=params,
                n_layers=n_layers,
                status_1q=st1,
                status_2q=st2,
                noise_strength_1q=noise_strength_1q,
                noise_strength_2q=noise_strength_2q,
                apply_two_qubit_noise=apply_two_qubit_noise,
                observable=observable,
            )
        )(st1_batch, st2_batch)
        valid = jnp.minimum(trajectory_batch_size, n_trajectories - start)
        mask = (jnp.arange(trajectory_batch_size) < valid).astype(scores.dtype)
        return acc + jnp.sum(scores * mask)

    total = jax.lax.fori_loop(0, num_batches, body, jnp.asarray(0.0, dtype=_DTYPE))
    return total / jnp.asarray(n_trajectories, dtype=_DTYPE)


def noisy_reuploading_sampled_score_streaming_status(
    key_measurement: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    status_1q: Array,
    status_2q: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    trajectory_batch_size: int = 16,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
    measurement_gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Streaming backend that treats each trajectory as one physical shot."""

    n_trajectories = int(status_1q.shape[0])
    trajectory_batch_size = int(trajectory_batch_size)
    if n_trajectories <= 0:
        raise ValueError("status_1q must contain at least one trajectory.")
    if trajectory_batch_size <= 0:
        raise ValueError("trajectory_batch_size must be positive.")
    if int(status_2q.shape[0]) != n_trajectories:
        raise ValueError("status_1q and status_2q must have the same trajectory count.")

    num_batches = (n_trajectories + trajectory_batch_size - 1) // trajectory_batch_size
    padded_n = num_batches * trajectory_batch_size
    pad_n = padded_n - n_trajectories
    status_1q_padded = jnp.pad(status_1q, ((0, pad_n), (0, 0), (0, 0)))
    status_2q_padded = jnp.pad(status_2q, ((0, pad_n), (0, 0), (0, 0)))

    def body(batch_idx, acc):
        start = batch_idx * trajectory_batch_size
        st1_batch = jax.lax.dynamic_slice_in_dim(
            status_1q_padded,
            start,
            trajectory_batch_size,
            axis=0,
        )
        st2_batch = jax.lax.dynamic_slice_in_dim(
            status_2q_padded,
            start,
            trajectory_batch_size,
            axis=0,
        )
        scores = jax.vmap(
            lambda st1, st2: noisy_reuploading_score_one_status(
                x,
                n_qubits=n_qubits,
                params=params,
                n_layers=n_layers,
                status_1q=st1,
                status_2q=st2,
                noise_strength_1q=noise_strength_1q,
                noise_strength_2q=noise_strength_2q,
                apply_two_qubit_noise=apply_two_qubit_noise,
                observable=observable,
            )
        )(st1_batch, st2_batch)
        outcomes = sample_single_pauli_outcomes(
            scores,
            jax.random.fold_in(key_measurement, batch_idx),
            gradient_mode=measurement_gradient_mode,
        )
        valid = jnp.minimum(trajectory_batch_size, n_trajectories - start)
        mask = (jnp.arange(trajectory_batch_size) < valid).astype(outcomes.dtype)
        return acc + jnp.sum(outcomes * mask)

    total = jax.lax.fori_loop(0, num_batches, body, jnp.asarray(0.0, dtype=_DTYPE))
    return total / jnp.asarray(n_trajectories, dtype=_DTYPE)


def _add_effective_shot_noise(score: Array, key: Array, n_shots_eff: int | None) -> Array:
    if n_shots_eff is None or int(n_shots_eff) <= 0:
        return score
    var = (1.0 - score**2) / float(n_shots_eff)
    return score + jnp.sqrt(jnp.abs(var)) * jax.random.normal(key, dtype=_DTYPE)


@partial(jax.jit, static_argnames=("n_qubits", "n_layers", "n_trajectories", "noise_strength_1q", "noise_strength_2q", "n_shots_eff", "apply_two_qubit_noise", "observable", "measurement_sampling_distribution", "measurement_gradient_mode"))
def sample_noisy_reuploading_score_vmap(
    key: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    n_trajectories: int,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    n_shots_eff: int | None = None,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Sample trajectories and finite-shot measurements.

    With ``measurement_sampling_distribution='binomial'``, each trajectory is
    treated as one physical shot: sample a fresh channel realization, then one
    Pauli outcome, and average the outcomes.  The legacy ``gaussian`` path keeps
    the old averaged-channel plus Gaussian effective-shot approximation.
    """

    key_status, key_shot = jax.random.split(key)
    status_1q, status_2q = make_reuploading_mc_statuses(
        key_status,
        n_trajectories=n_trajectories,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    if measurement_sampling_distribution == "binomial":
        return noisy_reuploading_sampled_score_vmap_status(
            key_shot,
            x,
            n_qubits=n_qubits,
            params=params,
            n_layers=n_layers,
            status_1q=status_1q,
            status_2q=status_2q,
            noise_strength_1q=noise_strength_1q,
            noise_strength_2q=noise_strength_2q,
            apply_two_qubit_noise=apply_two_qubit_noise,
            observable=observable,
            measurement_gradient_mode=measurement_gradient_mode,
        )

    score = noisy_reuploading_score_vmap_status(
        x,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
        status_1q=status_1q,
        status_2q=status_2q,
        noise_strength_1q=noise_strength_1q,
        noise_strength_2q=noise_strength_2q,
        apply_two_qubit_noise=apply_two_qubit_noise,
        observable=observable,
    )
    return jnp.clip(_add_effective_shot_noise(score, key_shot, n_shots_eff), -1.0, 1.0)


@partial(jax.jit, static_argnames=("n_qubits", "n_layers", "n_trajectories", "noise_strength_1q", "noise_strength_2q", "n_shots_eff", "trajectory_batch_size", "apply_two_qubit_noise", "observable", "measurement_sampling_distribution", "measurement_gradient_mode"))
def sample_noisy_reuploading_score_streaming(
    key: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    n_trajectories: int,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    n_shots_eff: int | None = None,
    trajectory_batch_size: int = 16,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
    measurement_sampling_distribution: str = "binomial",
    measurement_gradient_mode: str = "straight_through_expectation",
) -> Array:
    """Memory-friendlier trajectory sampler using streaming accumulation."""

    key_status, key_shot = jax.random.split(key)
    status_1q, status_2q = make_reuploading_mc_statuses(
        key_status,
        n_trajectories=n_trajectories,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    if measurement_sampling_distribution == "binomial":
        return noisy_reuploading_sampled_score_streaming_status(
            key_shot,
            x,
            n_qubits=n_qubits,
            params=params,
            n_layers=n_layers,
            status_1q=status_1q,
            status_2q=status_2q,
            noise_strength_1q=noise_strength_1q,
            noise_strength_2q=noise_strength_2q,
            trajectory_batch_size=trajectory_batch_size,
            apply_two_qubit_noise=apply_two_qubit_noise,
            observable=observable,
            measurement_gradient_mode=measurement_gradient_mode,
        )

    score = noisy_reuploading_score_streaming_status(
        x,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
        status_1q=status_1q,
        status_2q=status_2q,
        noise_strength_1q=noise_strength_1q,
        noise_strength_2q=noise_strength_2q,
        trajectory_batch_size=trajectory_batch_size,
        apply_two_qubit_noise=apply_two_qubit_noise,
        observable=observable,
    )
    return jnp.clip(_add_effective_shot_noise(score, key_shot, n_shots_eff), -1.0, 1.0)
