"""Explicit gatewise probabilistic error cancellation for strict-v5.

Each raw PEC sample independently chooses an inverse-depolarizing Pauli term
at every original noisy gate occurrence, propagates the resulting complete
auxiliary circuit with the strict-v5 density-matrix noise model, and draws one
fresh parity outcome.  Exact ideal expectations are never used by this forward
estimator.  They are used only by the separately declared simulation-side
straight-through derivative in :func:`explicit_pec_straight_through_loss_gradient`.

Correction Paulis are ideal classical-frame operations: they receive no extra
physical-noise channel.  This is the assumption under which the stored QPD
coefficients reconstruct the ideal gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from boosting_qml.losses.surrogates import PsiG
from boosting_qml.pqc.ansatz import DTYPE
from boosting_qml.pqc.density_matrix import (
    _CDTYPE,
    _apply_1q_depolarizing,
    _apply_2q_depolarizing,
    _apply_unitary,
    _cz_op,
    _identity,
    _observable_op,
    _pauli_x,
    _pauli_y,
    _pauli_z,
    _rx,
    _ry,
    _rz,
    _single_qubit_op,
    _two_qubit_pauli_op,
)
from boosting_qml.pqc.monte_carlo import _entangler_pairs
from boosting_qml.rng import stable_int_seed
from boosting_qml.training.finite_shot_parameter_shift import (
    _density_expectation_shift_tensor,
)


EXPLICIT_PEC_IMPLEMENTATION = "gatewise_qpd_density_matrix.one_shot.v1"
EXPLICIT_PEC_BATCHED_IMPLEMENTATION = "gatewise_qpd_density_matrix.grouped_tape.v2"
_PAULI_LABELS = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class DepolarizingQPD:
    support_size: int
    noise_strength: float
    coefficients: tuple[float, ...]
    probabilities: tuple[float, ...]
    signs: tuple[int, ...]
    norm: float


@dataclass(frozen=True)
class ExplicitPECTape:
    """Complete logical random tape for reproducible scalar/batched replay."""

    sample_ids: np.ndarray
    choices_1q: np.ndarray
    choices_2q: np.ndarray
    signs: np.ndarray
    measurement_uniforms: np.ndarray
    root_seed: int

    @property
    def count(self) -> int:
        return int(self.sample_ids.size)


def depolarizing_qpd(noise_strength: float, support_size: int) -> DepolarizingQPD:
    """Return the exact inverse-channel Pauli QPD used by the manuscript."""

    p, k = float(noise_strength), int(support_size)
    if k not in (1, 2):
        raise ValueError("strict-v5 explicit PEC supports one- and two-qubit gates.")
    if not math.isfinite(p) or not 0.0 <= p < 1.0:
        raise ValueError("noise_strength must be finite and lie in [0, 1).")
    cardinality = 4**k
    q_identity = (1.0 - p / cardinality) / (1.0 - p)
    q_nonidentity = -p / (cardinality * (1.0 - p))
    coeffs = (q_identity,) + (q_nonidentity,) * (cardinality - 1)
    norm = float(sum(abs(value) for value in coeffs))
    probabilities = tuple(float(abs(value) / norm) for value in coeffs)
    signs = tuple(1 if value >= 0.0 else -1 for value in coeffs)
    if not math.isclose(sum(coeffs), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise AssertionError("QPD coefficients do not sum to one.")
    return DepolarizingQPD(k, p, coeffs, probabilities, signs, norm)


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


def gate_occurrence_kinds(n_qubits: int, n_layers: int) -> tuple[str, ...]:
    """Return one entry per original modeled noise occurrence, in circuit order."""

    n, depth = int(n_qubits), int(n_layers)
    kinds: list[str] = []
    for _ in range(depth):
        kinds.extend(["1q"] * (3 * n))
        kinds.extend(["2q"] * ring_entangler_pair_count(n))
        kinds.extend(["1q"] * n)
    kinds.extend(["1q"] * (3 * n))
    return tuple(kinds)


@dataclass(frozen=True)
class ExplicitPECProtocol:
    n_qubits: int
    n_layers: int
    noise_strength_1q: float
    noise_strength_2q: float
    raw_samples_per_expectation: int
    apply_two_qubit_noise: bool = True
    chunk_size: int = 256
    represented_gate_kinds: tuple[str, ...] = ("1q", "2q")

    def __post_init__(self) -> None:
        if int(self.raw_samples_per_expectation) <= 0:
            raise ValueError("raw_samples_per_expectation must be positive.")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        validate_complete_representations(self)

    @property
    def qpd_1q(self) -> DepolarizingQPD:
        return depolarizing_qpd(self.noise_strength_1q, 1)

    @property
    def qpd_2q(self) -> DepolarizingQPD:
        return depolarizing_qpd(self.noise_strength_2q, 2)

    @property
    def gate_counts(self) -> tuple[int, int]:
        return legacy_reuploading_gate_counts(self.n_qubits, self.n_layers)

    @property
    def gamma(self) -> float:
        g1, g2 = self.gate_counts
        result = self.qpd_1q.norm**g1
        if self.apply_two_qubit_noise:
            result *= self.qpd_2q.norm**g2
        return float(result)

    @property
    def gamma_squared(self) -> float:
        return float(self.gamma**2)

    def to_dict(self) -> dict[str, Any]:
        g1, g2 = self.gate_counts
        return {
            "implementation_identity": EXPLICIT_PEC_IMPLEMENTATION,
            "n_qubits": int(self.n_qubits),
            "n_layers": int(self.n_layers),
            "noise_strength_1q": float(self.noise_strength_1q),
            "noise_strength_2q": float(self.noise_strength_2q),
            "raw_samples_per_expectation": int(self.raw_samples_per_expectation),
            "shots_per_sampled_circuit": 1,
            "one_qubit_gate_count_G1": g1,
            "two_qubit_gate_count_G2": g2,
            "one_qubit_coefficients": list(self.qpd_1q.coefficients),
            "two_qubit_coefficients": list(self.qpd_2q.coefficients),
            "one_qubit_sampling_probabilities": list(self.qpd_1q.probabilities),
            "two_qubit_sampling_probabilities": list(self.qpd_2q.probabilities),
            "one_qubit_qpd_norm": self.qpd_1q.norm,
            "two_qubit_qpd_norm": self.qpd_2q.norm,
            "coefficient_derived_circuit_gamma": self.gamma,
            "coefficient_derived_circuit_gamma_squared": self.gamma_squared,
            "correction_operations_receive_extra_noise": False,
            "represented_gate_kinds": list(self.represented_gate_kinds),
        }


def validate_complete_representations(protocol: ExplicitPECProtocol) -> None:
    required = {"1q"}
    if protocol.apply_two_qubit_noise:
        required.add("2q")
    missing = required.difference(protocol.represented_gate_kinds)
    if missing:
        raise ValueError(f"Missing PEC representations for original gate kinds: {sorted(missing)}")


def _one_qubit_correction_ops(qubit: int, n_qubits: int) -> jax.Array:
    return jnp.stack(
        [_single_qubit_op(gate, qubit, n_qubits) for gate in (_identity(), _pauli_x(), _pauli_y(), _pauli_z())]
    )


def _two_qubit_correction_ops(qa: int, qb: int, n_qubits: int) -> jax.Array:
    return jnp.stack(
        [_two_qubit_pauli_op(pa, pb, qa, qb, n_qubits) for pa in range(4) for pb in range(4)]
    )


def _correct_1q(rho: jax.Array, choice: jax.Array, qubit: int, n_qubits: int) -> jax.Array:
    return _apply_unitary(rho, _one_qubit_correction_ops(qubit, n_qubits)[choice])


def _correct_2q(
    rho: jax.Array, choice: jax.Array, qa: int, qb: int, n_qubits: int
) -> jax.Array:
    return _apply_unitary(rho, _two_qubit_correction_ops(qa, qb, n_qubits)[choice])


def apply_subsystem_depolarizing(
    rho: jax.Array,
    support: tuple[int, ...],
    n_qubits: int,
    noise_strength: float,
) -> jax.Array:
    """Apply ``(1-p)rho + p I_A/d_A tensor Tr_A(rho)`` on a gate support.

    The implementation permutes the selected, possibly noncontiguous, qubits
    to the front for both density-matrix indices, inserts the maximally mixed
    state on that support, then restores the original wire order.
    """

    p = float(noise_strength)
    if p <= 0.0:
        return rho
    n = int(n_qubits)
    selected = tuple(int(q) for q in support)
    if len(selected) not in (1, 2) or len(set(selected)) != len(selected):
        raise ValueError("depolarizing support must contain one or two distinct qubits")
    if any(q < 0 or q >= n for q in selected):
        raise ValueError("depolarizing support lies outside the register")
    rest = tuple(q for q in range(n) if q not in selected)
    row_order = selected + rest
    permutation = row_order + tuple(n + q for q in row_order)
    inverse = tuple(int(i) for i in np.argsort(permutation))
    tensor = jnp.asarray(rho, dtype=_CDTYPE).reshape((2,) * (2 * n))
    permuted = jnp.transpose(tensor, permutation)
    d_a, d_b = 2 ** len(selected), 2 ** len(rest)
    blocks = permuted.reshape(d_a, d_b, d_a, d_b)
    reduced = jnp.einsum("abad->bd", blocks)
    mixed = jnp.einsum(
        "ac,bd->abcd", jnp.eye(d_a, dtype=_CDTYPE) / d_a, reduced
    ).reshape((2,) * (2 * n))
    restored = jnp.transpose(mixed, inverse).reshape(rho.shape)
    return (1.0 - p) * rho + p * restored


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
def sampled_auxiliary_circuit_mean(
    x: jax.Array,
    params: jax.Array,
    choices_1q: jax.Array,
    choices_2q: jax.Array,
    *,
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
) -> jax.Array:
    """Exact mean of one explicitly selected noisy auxiliary circuit."""

    x = jnp.asarray(x, dtype=DTYPE).reshape(-1)
    params = jnp.asarray(params, dtype=DTYPE)
    dim = 2**int(n_qubits)
    psi0 = jnp.zeros((dim,), dtype=_CDTYPE).at[0].set(1.0 + 0.0j)
    rho = jnp.outer(psi0, jnp.conjugate(psi0))
    i1 = 0
    i2 = 0
    pairs = _entangler_pairs(int(n_qubits))

    for layer in range(int(n_layers)):
        lambda_l = params[4 * layer + 0]
        rotations = (
            (params[4 * layer + 1], _rx),
            (params[4 * layer + 2], _ry),
            (params[4 * layer + 3], _rz),
        )
        for angles, factory in rotations:
            for q in range(int(n_qubits)):
                rho = _apply_unitary(rho, _single_qubit_op(factory(angles[q]), q, int(n_qubits)))
                rho = apply_subsystem_depolarizing(
                    rho, (q,), int(n_qubits), float(noise_strength_1q)
                )
                rho = _correct_1q(rho, choices_1q[i1], q, int(n_qubits))
                i1 += 1
        for qa, qb in pairs:
            rho = _apply_unitary(rho, _cz_op(qa, qb, int(n_qubits)))
            if apply_two_qubit_noise:
                rho = apply_subsystem_depolarizing(
                    rho, (qa, qb), int(n_qubits), float(noise_strength_2q)
                )
                rho = _correct_2q(rho, choices_2q[i2], qa, qb, int(n_qubits))
                i2 += 1
        for q in range(int(n_qubits)):
            rho = _apply_unitary(
                rho, _single_qubit_op(_rx(x[q] * lambda_l[q]), q, int(n_qubits))
            )
            rho = apply_subsystem_depolarizing(
                rho, (q,), int(n_qubits), float(noise_strength_1q)
            )
            rho = _correct_1q(rho, choices_1q[i1], q, int(n_qubits))
            i1 += 1

    final_offset = 4 * int(n_layers)
    for angles, factory in (
        (params[final_offset + 0], _rx),
        (params[final_offset + 1], _ry),
        (params[final_offset + 2], _rz),
    ):
        for q in range(int(n_qubits)):
            rho = _apply_unitary(rho, _single_qubit_op(factory(angles[q]), q, int(n_qubits)))
            rho = apply_subsystem_depolarizing(
                rho, (q,), int(n_qubits), float(noise_strength_1q)
            )
            rho = _correct_1q(rho, choices_1q[i1], q, int(n_qubits))
            i1 += 1

    return jnp.real(jnp.trace(_observable_op(int(n_qubits), observable) @ rho)).astype(DTYPE)


def _draw_choices(
    rng: np.random.Generator, protocol: ExplicitPECProtocol, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g1, g2 = protocol.gate_counts
    one = rng.choice(4, size=(count, g1), p=protocol.qpd_1q.probabilities).astype(np.int32)
    if protocol.apply_two_qubit_noise:
        two = rng.choice(16, size=(count, g2), p=protocol.qpd_2q.probabilities).astype(np.int32)
    else:
        two = np.zeros((count, 0), dtype=np.int32)
    negative_terms = np.count_nonzero(one, axis=1)
    if two.shape[1]:
        negative_terms += np.count_nonzero(two, axis=1)
    signs = np.where(negative_terms % 2 == 0, 1.0, -1.0)
    return one, two, signs


def _counter_uniform_matrix(
    root_seed: int, sample_ids: np.ndarray, field_start: int, field_count: int
) -> np.ndarray:
    """Counter-based deterministic uniforms, stable under chunking/resumption."""

    if field_count == 0:
        return np.zeros((len(sample_ids), 0), dtype=np.float64)
    ids = np.asarray(sample_ids, dtype=np.uint64)[:, None]
    fields = np.arange(field_start, field_start + field_count, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        values = (
            np.uint64(int(root_seed) & ((1 << 64) - 1))
            + ids * np.uint64(0x9E3779B97F4A7C15)
            + fields * np.uint64(0xD1B54A32D192ED03)
        )
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return ((values >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))


def make_explicit_pec_tape(
    protocol: ExplicitPECProtocol,
    *,
    seed: int,
    start_sample_id: int = 0,
    count: int | None = None,
) -> ExplicitPECTape:
    """Pre-sample every QPD choice and measurement uniform by logical ID."""

    n_raw = int(protocol.raw_samples_per_expectation if count is None else count)
    if n_raw <= 0 or int(start_sample_id) < 0:
        raise ValueError("tape range must be nonempty and nonnegative")
    ids = np.arange(start_sample_id, start_sample_id + n_raw, dtype=np.uint64)
    g1, g2 = protocol.gate_counts
    u1 = _counter_uniform_matrix(seed, ids, 0, g1)
    one = np.searchsorted(np.cumsum(protocol.qpd_1q.probabilities), u1, side="right")
    if protocol.apply_two_qubit_noise:
        u2 = _counter_uniform_matrix(seed, ids, g1, g2)
        two = np.searchsorted(np.cumsum(protocol.qpd_2q.probabilities), u2, side="right")
    else:
        two = np.zeros((n_raw, 0), dtype=np.int64)
    negative = np.count_nonzero(one, axis=1) + np.count_nonzero(two, axis=1)
    signs = np.where(negative % 2 == 0, 1.0, -1.0)
    uniforms = _counter_uniform_matrix(seed, ids, g1 + g2, 1)[:, 0]
    return ExplicitPECTape(
        sample_ids=ids,
        choices_1q=np.asarray(one, dtype=np.int32),
        choices_2q=np.asarray(two, dtype=np.int32),
        signs=np.asarray(signs, dtype=np.float64),
        measurement_uniforms=np.asarray(uniforms, dtype=np.float64),
        root_seed=int(seed),
    )


@lru_cache(maxsize=32)
def _propagation_kernel(
    n_qubits: int,
    n_layers: int,
    noise_strength_1q: float,
    noise_strength_2q: float,
    apply_two_qubit_noise: bool,
    observable: str,
    propagation_batch_size: int,
):
    """Cache one compiled, fixed-shape propagation kernel per protocol shape."""

    def kernel(x, params, choices_1q, choices_2q):
        return jax.vmap(
            lambda c1, c2: sampled_auxiliary_circuit_mean(
                x,
                params,
                c1,
                c2,
                n_qubits=n_qubits,
                n_layers=n_layers,
                noise_strength_1q=noise_strength_1q,
                noise_strength_2q=noise_strength_2q,
                apply_two_qubit_noise=apply_two_qubit_noise,
                observable=observable,
            )
        )(choices_1q, choices_2q)

    return jax.jit(kernel)


def _finish_tape_replay(
    protocol: ExplicitPECProtocol,
    tape: ExplicitPECTape,
    means: np.ndarray,
    *,
    implementation: str,
    simulator_propagations: int,
    padded_simulator_propagations: int,
    backend_calls: int,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    means = np.asarray(means, dtype=np.float64)
    if means.shape != (tape.count,):
        raise ValueError("one conditional mean is required per logical raw sample")
    if np.any(~np.isfinite(means)) or np.any(np.abs(means) > 1.00001):
        raise FloatingPointError("auxiliary-circuit expectation left the physical interval")
    probabilities = np.clip((1.0 + means) / 2.0, 0.0, 1.0)
    parity = np.where(tape.measurement_uniforms < probabilities, 1.0, -1.0)
    contributions = protocol.gamma * tape.signs * parity
    combined = np.concatenate((tape.choices_1q, tape.choices_2q), axis=1)
    distinct = int(np.unique(combined, axis=0).shape[0])
    ledger = {
        "implementation_identity": implementation,
        "expectation_estimations": 1,
        "raw_physical_shots": tape.count,
        "sampled_circuit_instances": tape.count,
        "logical_sample_id_first": int(tape.sample_ids[0]),
        "logical_sample_id_last": int(tape.sample_ids[-1]),
        "distinct_sampled_qpd_paths": distinct,
        "simulator_propagations": int(simulator_propagations),
        "padded_simulator_propagations": int(padded_simulator_propagations),
        "backend_calls": int(backend_calls),
        "shots_per_sampled_circuit": 1,
        "correction_operations_receive_extra_noise": False,
        "exact_ideal_forward_accesses": 0,
        "gamma": protocol.gamma,
        "gamma_squared": protocol.gamma_squared,
    }
    trace = {
        "sample_ids": tape.sample_ids.copy(),
        "conditional_means": means,
        "measurement_uniforms": tape.measurement_uniforms.copy(),
        "parity_outcomes": parity,
        "contributions": contributions,
    }
    return float(np.mean(contributions)), ledger, trace


def replay_explicit_pec_tape_scalar(
    params: jax.Array,
    x: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    tape: ExplicitPECTape,
    observable: str = "global_z",
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Un-grouped correctness oracle using the same logical random tape."""

    means = []
    for c1, c2 in zip(tape.choices_1q, tape.choices_2q):
        value = sampled_auxiliary_circuit_mean(
            x,
            params,
            jnp.asarray(c1),
            jnp.asarray(c2),
            n_qubits=protocol.n_qubits,
            n_layers=protocol.n_layers,
            noise_strength_1q=protocol.noise_strength_1q,
            noise_strength_2q=protocol.noise_strength_2q,
            apply_two_qubit_noise=protocol.apply_two_qubit_noise,
            observable=observable,
        )
        means.append(float(jax.device_get(value)))
    return _finish_tape_replay(
        protocol,
        tape,
        np.asarray(means),
        implementation=EXPLICIT_PEC_IMPLEMENTATION + ".scalar_tape_oracle",
        simulator_propagations=tape.count,
        padded_simulator_propagations=0,
        backend_calls=tape.count,
    )


def replay_explicit_pec_tape_grouped(
    params: jax.Array,
    x: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    tape: ExplicitPECTape,
    propagation_batch_size: int = 128,
    observable: str = "global_z",
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate unique sampled paths once, retaining every physical shot."""

    batch = int(propagation_batch_size)
    if batch <= 0:
        raise ValueError("propagation_batch_size must be positive")
    combined = np.concatenate((tape.choices_1q, tape.choices_2q), axis=1)
    unique, inverse = np.unique(combined, axis=0, return_inverse=True)
    g1, _ = protocol.gate_counts
    unique_one = unique[:, :g1].astype(np.int32, copy=False)
    unique_two = unique[:, g1:].astype(np.int32, copy=False)
    kernel = _propagation_kernel(
        int(protocol.n_qubits),
        int(protocol.n_layers),
        float(protocol.noise_strength_1q),
        float(protocol.noise_strength_2q),
        bool(protocol.apply_two_qubit_noise),
        str(observable),
        batch,
    )
    values = []
    padded = 0
    calls = 0
    for start in range(0, len(unique), batch):
        size = min(batch, len(unique) - start)
        one = np.zeros((batch, unique_one.shape[1]), dtype=np.int32)
        two = np.zeros((batch, unique_two.shape[1]), dtype=np.int32)
        one[:size] = unique_one[start : start + size]
        two[:size] = unique_two[start : start + size]
        output = kernel(x, params, jnp.asarray(one), jnp.asarray(two))
        output = np.asarray(jax.device_get(output), dtype=np.float64)
        values.append(output[:size])
        padded += batch - size
        calls += 1
    unique_means = np.concatenate(values)
    means = unique_means[inverse]
    return _finish_tape_replay(
        protocol,
        tape,
        means,
        implementation=EXPLICIT_PEC_BATCHED_IMPLEMENTATION,
        simulator_propagations=len(unique),
        padded_simulator_propagations=padded,
        backend_calls=calls,
    )


def explicit_pec_expectation_grouped(
    params: jax.Array,
    x: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    seed: int,
    propagation_batch_size: int = 128,
    observable: str = "global_z",
) -> tuple[float, dict[str, Any]]:
    """Optimized explicit estimator; never reuses a random measurement outcome."""

    tape = make_explicit_pec_tape(protocol, seed=seed)
    estimate, ledger, _ = replay_explicit_pec_tape_grouped(
        params,
        x,
        protocol=protocol,
        tape=tape,
        propagation_batch_size=propagation_batch_size,
        observable=observable,
    )
    return estimate, ledger


def explicit_pec_expectation(
    params: jax.Array,
    x: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    seed: int,
    observable: str = "global_z",
) -> tuple[float, dict[str, Any]]:
    """Return one explicit PEC expectation and its physical execution ledger."""

    validate_complete_representations(protocol)
    rng = np.random.default_rng(int(seed))
    n_raw = int(protocol.raw_samples_per_expectation)
    contributions: list[np.ndarray] = []
    unique_paths: set[bytes] = set()
    for start in range(0, n_raw, int(protocol.chunk_size)):
        size = min(int(protocol.chunk_size), n_raw - start)
        one, two, signs = _draw_choices(rng, protocol, size)
        for row_one, row_two in zip(one, two):
            unique_paths.add(row_one.tobytes() + row_two.tobytes())
        means = jax.vmap(
            lambda c1, c2: sampled_auxiliary_circuit_mean(
                x,
                params,
                c1,
                c2,
                n_qubits=protocol.n_qubits,
                n_layers=protocol.n_layers,
                noise_strength_1q=protocol.noise_strength_1q,
                noise_strength_2q=protocol.noise_strength_2q,
                apply_two_qubit_noise=protocol.apply_two_qubit_noise,
                observable=observable,
            )
        )(jnp.asarray(one), jnp.asarray(two))
        means_np = np.asarray(jax.device_get(means), dtype=np.float64)
        if np.any(~np.isfinite(means_np)) or np.any(np.abs(means_np) > 1.00001):
            raise FloatingPointError("Auxiliary-circuit expectation left the physical interval.")
        probabilities = np.clip((1.0 + means_np) / 2.0, 0.0, 1.0)
        parity = np.where(rng.random(size) < probabilities, 1.0, -1.0)
        contributions.append(protocol.gamma * signs * parity)
    values = np.concatenate(contributions)
    return float(np.mean(values)), {
        "expectation_estimations": 1,
        "raw_physical_shots": n_raw,
        "sampled_circuit_instances": n_raw,
        "distinct_sampled_qpd_paths": len(unique_paths),
        "backend_calls": int(math.ceil(n_raw / int(protocol.chunk_size))),
        "shots_per_sampled_circuit": 1,
        "correction_operations_receive_extra_noise": False,
        "exact_ideal_forward_accesses": 0,
        "gamma": protocol.gamma,
        "gamma_squared": protocol.gamma_squared,
        "raw_estimate_min_possible": -protocol.gamma,
        "raw_estimate_max_possible": protocol.gamma,
    }


def explicit_pec_predict_batch(
    params: jax.Array,
    inputs: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    base_seed: int,
    stream: str,
    observable: str = "global_z",
) -> tuple[jax.Array, dict[str, Any]]:
    scores: list[float] = []
    ledgers: list[dict[str, Any]] = []
    for index, x in enumerate(np.asarray(inputs)):
        seed = stable_int_seed(EXPLICIT_PEC_IMPLEMENTATION, base_seed, stream, index)
        score, ledger = explicit_pec_expectation(
            params, jnp.asarray(x), protocol=protocol, seed=seed, observable=observable
        )
        scores.append(score)
        ledgers.append(ledger)
    return jnp.asarray(scores, dtype=DTYPE), {
        "expectation_estimations": len(scores),
        "raw_physical_shots": sum(int(row["raw_physical_shots"]) for row in ledgers),
        "sampled_circuit_instances": sum(int(row["sampled_circuit_instances"]) for row in ledgers),
        "distinct_sampled_qpd_paths_sum": sum(int(row["distinct_sampled_qpd_paths"]) for row in ledgers),
        "backend_calls": sum(int(row["backend_calls"]) for row in ledgers),
        "exact_ideal_forward_accesses": 0,
        "implementation_identity": EXPLICIT_PEC_IMPLEMENTATION,
    }


def explicit_pec_predict_batch_grouped(
    params: jax.Array,
    inputs: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    base_seed: int,
    stream: str,
    propagation_batch_size: int = 128,
    observable: str = "global_z",
) -> tuple[jax.Array, dict[str, Any]]:
    """Predict examples with logical-ID tapes and exact duplicate grouping."""

    scores: list[float] = []
    ledgers: list[dict[str, Any]] = []
    for index, x in enumerate(np.asarray(inputs)):
        seed = stable_int_seed(EXPLICIT_PEC_BATCHED_IMPLEMENTATION, base_seed, stream, index)
        score, ledger = explicit_pec_expectation_grouped(
            params,
            jnp.asarray(x),
            protocol=protocol,
            seed=seed,
            propagation_batch_size=propagation_batch_size,
            observable=observable,
        )
        scores.append(score)
        ledgers.append(ledger)
    return jnp.asarray(scores, dtype=DTYPE), {
        "expectation_estimations": len(scores),
        "raw_physical_shots": sum(int(row["raw_physical_shots"]) for row in ledgers),
        "sampled_circuit_instances": sum(
            int(row["sampled_circuit_instances"]) for row in ledgers
        ),
        "distinct_sampled_qpd_paths_sum": sum(
            int(row["distinct_sampled_qpd_paths"]) for row in ledgers
        ),
        "simulator_propagations": sum(int(row["simulator_propagations"]) for row in ledgers),
        "padded_simulator_propagations": sum(
            int(row["padded_simulator_propagations"]) for row in ledgers
        ),
        "backend_calls": sum(int(row["backend_calls"]) for row in ledgers),
        "exact_ideal_forward_accesses": 0,
        "implementation_identity": EXPLICIT_PEC_BATCHED_IMPLEMENTATION,
    }


def _ideal_scores(
    params: jax.Array,
    inputs: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    observable: str,
) -> jax.Array:
    return _density_expectation_shift_tensor(
        jnp.broadcast_to(params, (1, int(inputs.shape[0])) + tuple(params.shape)),
        inputs,
        n_qubits=protocol.n_qubits,
        n_layers=protocol.n_layers,
        noise_strength_1q=0.0,
        noise_strength_2q=0.0,
        apply_two_qubit_noise=protocol.apply_two_qubit_noise,
        observable=observable,
    )[0]


def explicit_pec_straight_through_loss_gradient(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    base_seed: int,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    observable: str = "global_z",
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Preserve strict-v5's simulator-assisted straight-through derivative."""

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    labels = jnp.asarray(labels, dtype=DTYPE)
    sampled, ledger = explicit_pec_predict_batch(
        params,
        inputs,
        protocol=protocol,
        base_seed=base_seed,
        stream="training_forward",
        observable=observable,
    )
    ideal_function = lambda current_params: _ideal_scores(
        current_params, inputs, protocol=protocol, observable=observable
    )
    _, ideal_pullback = jax.vjp(ideal_function, params)

    def objective(scores: jax.Array) -> jax.Array:
        losses = PsiG(labels * scores, sigma)
        if weights is None:
            return jnp.mean(losses)
        normalized = jnp.asarray(weights, dtype=losses.dtype)
        normalized = normalized / jnp.sum(normalized)
        return jnp.sum(normalized * losses)

    loss, score_cotangent = jax.value_and_grad(objective)(sampled)
    gradients = ideal_pullback(score_cotangent)[0]
    ledger = dict(ledger)
    ledger.update(
        {
            "gradient_semantics": "loss_derivative_at_explicit_pec_score_times_exact_ideal_mean_vjp",
            "parameter_shift_circuits_executed": 0,
            "gradient_raw_physical_shots": 0,
            "exact_ideal_derivative_evaluations": int(inputs.shape[0]),
        }
    )
    return gradients, loss, sampled, ledger


def explicit_pec_grouped_straight_through_loss_gradient(
    params: jax.Array,
    inputs: jax.Array,
    labels: jax.Array,
    *,
    protocol: ExplicitPECProtocol,
    base_seed: int,
    weights: jax.Array | None = None,
    sigma: float = 0.5,
    propagation_batch_size: int = 128,
    observable: str = "global_z",
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Grouped forward with the unchanged simulator-assisted backward rule."""

    params = jnp.asarray(params, dtype=DTYPE)
    inputs = jnp.asarray(inputs, dtype=DTYPE)
    labels = jnp.asarray(labels, dtype=DTYPE)
    sampled, ledger = explicit_pec_predict_batch_grouped(
        params,
        inputs,
        protocol=protocol,
        base_seed=base_seed,
        stream="training_forward",
        propagation_batch_size=propagation_batch_size,
        observable=observable,
    )
    ideal_function = lambda current_params: _ideal_scores(
        current_params, inputs, protocol=protocol, observable=observable
    )
    _, ideal_pullback = jax.vjp(ideal_function, params)

    def objective(scores: jax.Array) -> jax.Array:
        losses = PsiG(labels * scores, sigma)
        if weights is None:
            return jnp.mean(losses)
        normalized = jnp.asarray(weights, dtype=losses.dtype)
        normalized = normalized / jnp.sum(normalized)
        return jnp.sum(normalized * losses)

    loss, score_cotangent = jax.value_and_grad(objective)(sampled)
    gradients = ideal_pullback(score_cotangent)[0]
    ledger = dict(ledger)
    ledger.update(
        {
            "gradient_semantics": (
                "loss_derivative_at_explicit_pec_score_times_exact_ideal_mean_vjp"
            ),
            "parameter_shift_circuits_executed": 0,
            "gradient_raw_physical_shots": 0,
            "exact_ideal_derivative_evaluations": int(inputs.shape[0]),
        }
    )
    return gradients, loss, sampled, ledger


def mitiq_norm_crosscheck(qpd: DepolarizingQPD) -> dict[str, Any]:
    """Cross-check the manual representation with pinned Mitiq's container API."""

    import cirq
    from mitiq.pec import NoisyOperation, OperationRepresentation

    qubits = cirq.LineQubit.range(qpd.support_size)
    identity = cirq.Circuit(cirq.IdentityGate(qpd.support_size).on(*qubits))
    noisy_operations = []
    for index in range(4**qpd.support_size):
        circuit = identity.copy()
        digits = []
        value = index
        for _ in range(qpd.support_size):
            digits.append(value % 4)
            value //= 4
        pauli_gates = (cirq.I, cirq.X, cirq.Y, cirq.Z)
        for qubit, digit in zip(reversed(qubits), digits):
            circuit.append(pauli_gates[digit].on(qubit))
        noisy_operations.append(NoisyOperation(circuit))
    representation = OperationRepresentation(identity, noisy_operations, qpd.coefficients)
    return {
        "mitiq_version_checked": "0.48.1",
        "mitiq_representation_norm": float(representation.norm),
        "coefficient_norm": qpd.norm,
        "absolute_difference": abs(float(representation.norm) - qpd.norm),
    }


__all__ = [
    "EXPLICIT_PEC_BATCHED_IMPLEMENTATION",
    "EXPLICIT_PEC_IMPLEMENTATION",
    "DepolarizingQPD",
    "ExplicitPECTape",
    "ExplicitPECProtocol",
    "apply_subsystem_depolarizing",
    "depolarizing_qpd",
    "explicit_pec_expectation",
    "explicit_pec_expectation_grouped",
    "explicit_pec_grouped_straight_through_loss_gradient",
    "explicit_pec_predict_batch",
    "explicit_pec_predict_batch_grouped",
    "explicit_pec_straight_through_loss_gradient",
    "gate_occurrence_kinds",
    "legacy_reuploading_gate_counts",
    "make_explicit_pec_tape",
    "mitiq_norm_crosscheck",
    "replay_explicit_pec_tape_grouped",
    "replay_explicit_pec_tape_scalar",
    "sampled_auxiliary_circuit_mean",
    "validate_complete_representations",
]
