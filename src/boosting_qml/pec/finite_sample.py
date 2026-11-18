"""Finite-sample PEC estimators for legacy reuploading PQCs."""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy_compat import patch_numpy_for_tensorcircuit

patch_numpy_for_tensorcircuit()

import tensorcircuit as tc

from boosting_qml.pqc.monte_carlo import (
    _TWO_QUBIT_PAULIS,
    _apply_one_qubit_pauli_state,
    _apply_two_qubit_depolarizing_status_state,
    _entangler_pairs,
    _validate_legacy_inputs,
    make_reuploading_mc_statuses,
    reuploading_one_qubit_status_shape,
    reuploading_two_qubit_status_shape,
)

K = tc.set_backend("jax")

Array = jax.Array
_DTYPE = jnp.float32


def inverse_depolarizing_l1_norm_1q(noise_strength: float) -> float:
    """Return the one-qubit inverse-depolarizing quasi-probability L1 norm."""

    epsilon = float(noise_strength)
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"noise_strength must lie in [0, 1), got {epsilon}.")
    if epsilon == 0.0:
        return 1.0
    return float((1.0 + epsilon / 2.0) / (1.0 - epsilon))


def inverse_depolarizing_l1_norm_2q(noise_strength: float) -> float:
    """Return the two-qubit inverse-depolarizing quasi-probability L1 norm."""

    epsilon = float(noise_strength)
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"noise_strength must lie in [0, 1), got {epsilon}.")
    if epsilon == 0.0:
        return 1.0
    return float((1.0 + 7.0 * epsilon / 8.0) / (1.0 - epsilon))


def _inverse_1q_code_and_sign(status: Array, noise_strength: float) -> tuple[Array, Array]:
    if float(noise_strength) <= 0.0:
        return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1.0, dtype=_DTYPE)

    epsilon = float(noise_strength)
    q_identity_abs = (1.0 - epsilon / 4.0) / (1.0 - epsilon)
    q_non_identity_abs = epsilon / (4.0 * (1.0 - epsilon))
    gamma = q_identity_abs + 3.0 * q_non_identity_abs
    p_identity = q_identity_abs / gamma
    p_each = q_non_identity_abs / gamma
    non_identity = 1 + jnp.floor((jnp.asarray(status) - p_identity) / p_each).astype(jnp.int32)
    non_identity = jnp.clip(non_identity, 1, 3)
    code = jnp.where(jnp.asarray(status) < p_identity, 0, non_identity)
    sign = jnp.where(code == 0, 1.0, -1.0).astype(_DTYPE)
    return code.astype(jnp.int32), sign


def _inverse_2q_code_and_sign(status: Array, noise_strength: float) -> tuple[Array, Array]:
    if float(noise_strength) <= 0.0:
        return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1.0, dtype=_DTYPE)

    epsilon = float(noise_strength)
    q_identity_abs = (1.0 - epsilon / 16.0) / (1.0 - epsilon)
    q_non_identity_abs = epsilon / (16.0 * (1.0 - epsilon))
    gamma = q_identity_abs + 15.0 * q_non_identity_abs
    p_identity = q_identity_abs / gamma
    p_each = q_non_identity_abs / gamma
    non_identity = 1 + jnp.floor((jnp.asarray(status) - p_identity) / p_each).astype(jnp.int32)
    non_identity = jnp.clip(non_identity, 1, 15)
    code = jnp.where(jnp.asarray(status) < p_identity, 0, non_identity)
    sign = jnp.where(code == 0, 1.0, -1.0).astype(_DTYPE)
    return code.astype(jnp.int32), sign


def _apply_two_qubit_pauli_code_state(
    state: Array,
    qa: int,
    qb: int,
    pauli_code: Array,
    *,
    n_qubits: int,
) -> Array:
    branches = []
    for pa, pb in _TWO_QUBIT_PAULIS:
        def _branch(s, pa=pa, pb=pb):
            s = _apply_one_qubit_pauli_state(s, int(qa), int(pa), int(n_qubits))
            return _apply_one_qubit_pauli_state(s, int(qb), int(pb), int(n_qubits))

        branches.append(_branch)
    return jax.lax.switch(pauli_code.astype(jnp.int32), branches, state)


def _apply_one_qubit_pauli_code_state(
    state: Array,
    qubit: int,
    pauli_code: Array,
    *,
    n_qubits: int,
) -> Array:
    branches = []
    for pauli in range(4):
        def _branch(s, pauli=pauli):
            return _apply_one_qubit_pauli_state(s, int(qubit), int(pauli), int(n_qubits))

        branches.append(_branch)
    return jax.lax.switch(pauli_code.astype(jnp.int32), branches, state)


def make_reuploading_pec_statuses(
    key: Array,
    *,
    n_samples: int,
    n_qubits: int,
    n_layers: int,
    dtype: jnp.dtype = _DTYPE,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sample physical-noise, inverse-channel, and measurement statuses."""

    key_phys, key_inv_1q, key_inv_2q, key_measure = jax.random.split(key, 4)
    physical_1q, physical_2q = make_reuploading_mc_statuses(
        key_phys,
        n_trajectories=int(n_samples),
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        dtype=dtype,
    )
    inverse_1q = jax.random.uniform(
        key_inv_1q,
        shape=(int(n_samples),) + reuploading_one_qubit_status_shape(n_qubits, n_layers),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    inverse_2q = jax.random.uniform(
        key_inv_2q,
        shape=(int(n_samples),) + reuploading_two_qubit_status_shape(n_qubits, n_layers),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    measurement = jax.random.uniform(
        key_measure,
        shape=(int(n_samples),),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    )
    return physical_1q, physical_2q, inverse_1q, inverse_2q, measurement


def pec_reuploading_single_shot_estimator(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    physical_status_1q: Array,
    physical_status_2q: Array,
    inverse_status_1q: Array,
    inverse_status_2q: Array,
    measurement_status: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Return one signed, scaled PEC single-shot estimator."""

    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    x = jnp.asarray(x, dtype=_DTYPE).reshape(-1)
    params = jnp.asarray(params, dtype=_DTYPE)
    physical_status_1q = jnp.asarray(physical_status_1q, dtype=_DTYPE)
    physical_status_2q = jnp.asarray(physical_status_2q, dtype=_DTYPE)
    inverse_status_1q = jnp.asarray(inverse_status_1q, dtype=_DTYPE)
    inverse_status_2q = jnp.asarray(inverse_status_2q, dtype=_DTYPE)
    _validate_legacy_inputs(
        x,
        params,
        physical_status_1q,
        physical_status_2q,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    _validate_legacy_inputs(
        x,
        params,
        inverse_status_1q,
        inverse_status_2q,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    px1 = py1 = pz1 = float(noise_strength_1q) / 4.0
    p2_each = float(noise_strength_2q) / 16.0
    pairs = _entangler_pairs(n_qubits)
    gamma_1q = inverse_depolarizing_l1_norm_1q(float(noise_strength_1q))
    gamma_2q = inverse_depolarizing_l1_norm_2q(float(noise_strength_2q))
    gamma_total = gamma_1q ** ((4 * n_layers + 3) * n_qubits)
    if apply_two_qubit_noise:
        gamma_total *= gamma_2q ** (n_layers * len(pairs))
    gamma_total = jnp.asarray(gamma_total, dtype=_DTYPE)

    sign = jnp.asarray(1.0, dtype=_DTYPE)
    s = tc.Circuit(n_qubits).state()

    def _apply_1q_after_gate(circuit, state_sign, gate_row: int, qubit: int):
        state, current_sign = state_sign
        if float(noise_strength_1q) > 0.0:
            circuit.depolarizing(
                qubit,
                px=px1,
                py=py1,
                pz=pz1,
                status=physical_status_1q[gate_row, qubit],
            )
        state = circuit.state()
        code, gate_sign = _inverse_1q_code_and_sign(inverse_status_1q[gate_row, qubit], float(noise_strength_1q))
        state = _apply_one_qubit_pauli_code_state(state, int(qubit), code, n_qubits=n_qubits)
        return state, (current_sign * gate_sign).astype(_DTYPE)

    for layer in range(n_layers):
        lambda_l = params[4 * layer + 0]
        rx_l = params[4 * layer + 1]
        ry_l = params[4 * layer + 2]
        rz_l = params[4 * layer + 3]
        c = tc.Circuit(n_qubits, inputs=s)

        for q in range(n_qubits):
            c.rx(q, theta=rx_l[q])
            s, sign = _apply_1q_after_gate(c, (s, sign), 4 * layer + 0, q)
            c = tc.Circuit(n_qubits, inputs=s)
            c.ry(q, theta=ry_l[q])
            s, sign = _apply_1q_after_gate(c, (s, sign), 4 * layer + 1, q)
            c = tc.Circuit(n_qubits, inputs=s)
            c.rz(q, theta=rz_l[q])
            s, sign = _apply_1q_after_gate(c, (s, sign), 4 * layer + 2, q)
            c = tc.Circuit(n_qubits, inputs=s)

        c = tc.Circuit(n_qubits, inputs=s)
        for pair_idx, (qa, qb) in enumerate(pairs):
            c.cz(qa, qb)
            s = c.state()
            if apply_two_qubit_noise and float(noise_strength_2q) > 0.0:
                s = _apply_two_qubit_depolarizing_status_state(
                    s,
                    qa,
                    qb,
                    physical_status_2q[layer, pair_idx],
                    n_qubits=n_qubits,
                    p_each=p2_each,
                )
            if apply_two_qubit_noise:
                code, gate_sign = _inverse_2q_code_and_sign(
                    inverse_status_2q[layer, pair_idx],
                    float(noise_strength_2q),
                )
                s = _apply_two_qubit_pauli_code_state(s, qa, qb, code, n_qubits=n_qubits)
                sign = (sign * gate_sign).astype(_DTYPE)
            c = tc.Circuit(n_qubits, inputs=s)

        for q in range(n_qubits):
            c.rx(q, theta=x[q] * lambda_l[q])
            s, sign = _apply_1q_after_gate(c, (s, sign), 4 * layer + 3, q)
            c = tc.Circuit(n_qubits, inputs=s)

    final_offset = 4 * n_layers
    c = tc.Circuit(n_qubits, inputs=s)
    for q in range(n_qubits):
        c.rx(q, theta=params[final_offset + 0, q])
        s, sign = _apply_1q_after_gate(c, (s, sign), final_offset + 0, q)
        c = tc.Circuit(n_qubits, inputs=s)
        c.ry(q, theta=params[final_offset + 1, q])
        s, sign = _apply_1q_after_gate(c, (s, sign), final_offset + 1, q)
        c = tc.Circuit(n_qubits, inputs=s)
        c.rz(q, theta=params[final_offset + 2, q])
        s, sign = _apply_1q_after_gate(c, (s, sign), final_offset + 2, q)
        c = tc.Circuit(n_qubits, inputs=s)

    if observable == "global_z":
        expectation = c.expectation_ps(z=np.arange(n_qubits))
    elif observable == "z0":
        expectation = c.expectation_ps(z=[0])
    else:
        raise ValueError(f"Unsupported observable '{observable}'. Expected 'global_z' or 'z0'.")
    expectation = jnp.real(expectation)
    outcome = jnp.where(jnp.asarray(measurement_status) < (1.0 + expectation) / 2.0, 1.0, -1.0)
    return gamma_total * sign * outcome.astype(_DTYPE)


@partial(
    jax.jit,
    static_argnames=(
        "n_qubits",
        "n_layers",
        "noise_strength_1q",
        "noise_strength_2q",
        "apply_two_qubit_noise",
        "observable",
    ),
)
def pec_reuploading_estimate_from_statuses(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    physical_status_1q: Array,
    physical_status_2q: Array,
    inverse_status_1q: Array,
    inverse_status_2q: Array,
    measurement_status: Array,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Average finite-sample PEC single-shot estimators for one input."""

    estimators = jax.vmap(
        lambda pst1, pst2, ist1, ist2, mst: pec_reuploading_single_shot_estimator(
            x,
            n_qubits=n_qubits,
            params=params,
            n_layers=n_layers,
            physical_status_1q=pst1,
            physical_status_2q=pst2,
            inverse_status_1q=ist1,
            inverse_status_2q=ist2,
            measurement_status=mst,
            noise_strength_1q=noise_strength_1q,
            noise_strength_2q=noise_strength_2q,
            apply_two_qubit_noise=apply_two_qubit_noise,
            observable=observable,
        )
    )(physical_status_1q, physical_status_2q, inverse_status_1q, inverse_status_2q, measurement_status)
    return jnp.mean(estimators)


def finite_sample_pec_score(
    key: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    n_samples: int,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Estimate one PEC-mitigated expectation value from raw single-shot samples."""

    statuses = make_reuploading_pec_statuses(
        key,
        n_samples=int(n_samples),
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
    )
    return pec_reuploading_estimate_from_statuses(
        x,
        n_qubits=int(n_qubits),
        params=jnp.asarray(params, dtype=_DTYPE),
        n_layers=int(n_layers),
        physical_status_1q=statuses[0],
        physical_status_2q=statuses[1],
        inverse_status_1q=statuses[2],
        inverse_status_2q=statuses[3],
        measurement_status=statuses[4],
        noise_strength_1q=float(noise_strength_1q),
        noise_strength_2q=float(noise_strength_2q),
        apply_two_qubit_noise=bool(apply_two_qubit_noise),
        observable=observable,
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_qubits",
        "n_layers",
        "n_samples",
        "noise_strength_1q",
        "noise_strength_2q",
        "apply_two_qubit_noise",
        "observable",
    ),
)
def finite_sample_pec_scores_batch(
    key: Array,
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    n_samples: int,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Estimate PEC-mitigated expectation values for a batch of inputs."""

    x = jnp.asarray(x, dtype=_DTYPE)
    keys = jax.random.split(key, int(x.shape[0]))

    def _score_one(sample_key, x_value):
        statuses = make_reuploading_pec_statuses(
            sample_key,
            n_samples=int(n_samples),
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
        )
        return pec_reuploading_estimate_from_statuses(
            x_value,
            n_qubits=int(n_qubits),
            params=jnp.asarray(params, dtype=_DTYPE),
            n_layers=int(n_layers),
            physical_status_1q=statuses[0],
            physical_status_2q=statuses[1],
            inverse_status_1q=statuses[2],
            inverse_status_2q=statuses[3],
            measurement_status=statuses[4],
            noise_strength_1q=float(noise_strength_1q),
            noise_strength_2q=float(noise_strength_2q),
            apply_two_qubit_noise=bool(apply_two_qubit_noise),
            observable=observable,
        )

    return jax.vmap(_score_one)(keys, x)


__all__ = [
    "finite_sample_pec_score",
    "finite_sample_pec_scores_batch",
    "inverse_depolarizing_l1_norm_1q",
    "inverse_depolarizing_l1_norm_2q",
    "make_reuploading_pec_statuses",
    "pec_reuploading_estimate_from_statuses",
    "pec_reuploading_single_shot_estimator",
]
