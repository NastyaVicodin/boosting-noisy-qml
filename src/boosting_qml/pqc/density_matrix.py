"""Exact density-matrix simulator for legacy reuploading depolarizing PQCs."""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from boosting_qml.pqc.ansatz import DTYPE
from boosting_qml.pqc.monte_carlo import _TWO_QUBIT_NON_ID_PAULIS, _entangler_pairs

Array = jax.Array
_CDTYPE = jnp.complex64


def _identity() -> Array:
    return jnp.eye(2, dtype=_CDTYPE)


def _pauli_x() -> Array:
    return jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=_CDTYPE)


def _pauli_y() -> Array:
    return jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=_CDTYPE)


def _pauli_z() -> Array:
    return jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=_CDTYPE)


def _rx(theta: Array) -> Array:
    half = jnp.asarray(theta, dtype=DTYPE) / 2.0
    c = jnp.cos(half).astype(_CDTYPE)
    s = jnp.sin(half).astype(_CDTYPE)
    return jnp.asarray([[c, -1.0j * s], [-1.0j * s, c]], dtype=_CDTYPE)


def _ry(theta: Array) -> Array:
    half = jnp.asarray(theta, dtype=DTYPE) / 2.0
    c = jnp.cos(half).astype(_CDTYPE)
    s = jnp.sin(half).astype(_CDTYPE)
    return jnp.asarray([[c, -s], [s, c]], dtype=_CDTYPE)


def _rz(theta: Array) -> Array:
    half = jnp.asarray(theta, dtype=DTYPE) / 2.0
    return jnp.asarray(
        [[jnp.exp(-1.0j * half), 0.0], [0.0, jnp.exp(1.0j * half)]],
        dtype=_CDTYPE,
    )


def _kron_all(mats: list[Array]) -> Array:
    out = mats[0]
    for mat in mats[1:]:
        out = jnp.kron(out, mat)
    return out


def _single_qubit_op(gate: Array, qubit: int, n_qubits: int) -> Array:
    return _kron_all([gate if q == int(qubit) else _identity() for q in range(int(n_qubits))])


def _two_qubit_pauli_op(pa: int, pb: int, qa: int, qb: int, n_qubits: int) -> Array:
    paulis = (_identity(), _pauli_x(), _pauli_y(), _pauli_z())
    return _kron_all(
        [
            paulis[int(pa)] if q == int(qa) else paulis[int(pb)] if q == int(qb) else _identity()
            for q in range(int(n_qubits))
        ]
    )


def _cz_op(qa: int, qb: int, n_qubits: int) -> Array:
    n_qubits = int(n_qubits)
    dim = 2**n_qubits
    diag = []
    for idx in range(dim):
        bit_a = (idx >> (n_qubits - 1 - int(qa))) & 1
        bit_b = (idx >> (n_qubits - 1 - int(qb))) & 1
        diag.append(-1.0 if bit_a and bit_b else 1.0)
    return jnp.diag(jnp.asarray(diag, dtype=_CDTYPE))


def _apply_unitary(rho: Array, unitary: Array) -> Array:
    return unitary @ rho @ jnp.conjugate(unitary.T)


def _apply_1q_depolarizing(rho: Array, qubit: int, n_qubits: int, epsilon: float) -> Array:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        return rho
    p_each = epsilon / 4.0
    updated = (1.0 - 3.0 * p_each) * rho
    for pauli in (_pauli_x(), _pauli_y(), _pauli_z()):
        op = _single_qubit_op(pauli, int(qubit), int(n_qubits))
        updated = updated + p_each * _apply_unitary(rho, op)
    return updated


def _apply_2q_depolarizing(
    rho: Array,
    qa: int,
    qb: int,
    n_qubits: int,
    epsilon: float,
) -> Array:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        return rho
    p_each = epsilon / 16.0
    updated = (1.0 - 15.0 * p_each) * rho
    for pa, pb in _TWO_QUBIT_NON_ID_PAULIS:
        op = _two_qubit_pauli_op(pa, pb, int(qa), int(qb), int(n_qubits))
        updated = updated + p_each * _apply_unitary(rho, op)
    return updated


def _observable_op(n_qubits: int, observable: str) -> Array:
    if observable == "global_z":
        return _kron_all([_pauli_z() for _ in range(int(n_qubits))])
    if observable == "z0":
        return _single_qubit_op(_pauli_z(), 0, int(n_qubits))
    raise ValueError(f"Unsupported observable '{observable}'. Expected 'global_z' or 'z0'.")


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
def exact_noisy_reuploading_density_score(
    x: Array,
    *,
    n_qubits: int,
    params: Array,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float = 0.0,
    apply_two_qubit_noise: bool = True,
    observable: Literal["global_z", "z0"] = "global_z",
) -> Array:
    """Evaluate the exact mixed-state depolarizing-channel expectation."""

    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}.")
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")

    x = jnp.asarray(x, dtype=DTYPE).reshape(-1)
    params = jnp.asarray(params, dtype=DTYPE)
    dim = 2**n_qubits
    psi0 = jnp.zeros((dim,), dtype=_CDTYPE).at[0].set(1.0 + 0.0j)
    rho = jnp.outer(psi0, jnp.conjugate(psi0))

    pairs = _entangler_pairs(n_qubits)
    for layer in range(n_layers):
        lambda_l = params[4 * layer + 0]
        rx_l = params[4 * layer + 1]
        ry_l = params[4 * layer + 2]
        rz_l = params[4 * layer + 3]

        for q in range(n_qubits):
            rho = _apply_unitary(rho, _single_qubit_op(_rx(rx_l[q]), q, n_qubits))
            rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))
            rho = _apply_unitary(rho, _single_qubit_op(_ry(ry_l[q]), q, n_qubits))
            rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))
            rho = _apply_unitary(rho, _single_qubit_op(_rz(rz_l[q]), q, n_qubits))
            rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))

        for qa, qb in pairs:
            rho = _apply_unitary(rho, _cz_op(qa, qb, n_qubits))
            if apply_two_qubit_noise:
                rho = _apply_2q_depolarizing(
                    rho,
                    qa,
                    qb,
                    n_qubits,
                    float(noise_strength_2q),
                )

        for q in range(n_qubits):
            rho = _apply_unitary(rho, _single_qubit_op(_rx(x[q] * lambda_l[q]), q, n_qubits))
            rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))

    final_offset = 4 * n_layers
    for q in range(n_qubits):
        rho = _apply_unitary(rho, _single_qubit_op(_rx(params[final_offset + 0, q]), q, n_qubits))
        rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))
        rho = _apply_unitary(rho, _single_qubit_op(_ry(params[final_offset + 1, q]), q, n_qubits))
        rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))
        rho = _apply_unitary(rho, _single_qubit_op(_rz(params[final_offset + 2, q]), q, n_qubits))
        rho = _apply_1q_depolarizing(rho, q, n_qubits, float(noise_strength_1q))

    obs = _observable_op(n_qubits, str(observable))
    return jnp.real(jnp.trace(obs @ rho)).astype(DTYPE)


__all__ = ["exact_noisy_reuploading_density_score"]
