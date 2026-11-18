"""Shared PQC ansatz and parameter initialization utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int
from numpy_compat import patch_numpy_for_tensorcircuit

patch_numpy_for_tensorcircuit()

import tensorcircuit as tc

DTYPE = jnp.float32
K = tc.set_backend("jax")
tc.set_dtype("complex64")

def build_reuploading_circuit_layer_scan(
    X: jax.Array,
    n_qubits: Int,
    params: jax.Array,
    n_layers: Int,
):
    """Build the reuploading circuit via per-layer state evolution with ``K.scan``.

    This mirrors ``quantum_teacher_student_dataset.reuploading_circuit`` exactly:
      - per layer: RX/RY/RZ variational -> ring CZ entangling -> RX data reupload
      - final: RX/RY/RZ variational block
    with parameter layout ``(4 * n_layers + 3, n_qubits)``.
    """
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)

    X = jnp.asarray(X, dtype=DTYPE).reshape(-1)
    params = jnp.asarray(params, dtype=DTYPE)

    expected_shape = (4 * n_layers + 3, n_qubits)
    if params.shape != expected_shape:
        raise ValueError(f"Expected params shape {expected_shape}, got {tuple(params.shape)}.")
    if X.shape[0] != n_qubits:
        raise ValueError(f"Expected X.shape[0] == {n_qubits}, got {X.shape[0]}.")

    # Group each layer into [lambda, rx, ry, rz] for efficient scan traversal.
    layer_params = jnp.stack(
        [
            params[0 : 4 * n_layers : 4, :],
            params[1 : 4 * n_layers : 4, :],
            params[2 : 4 * n_layers : 4, :],
            params[3 : 4 * n_layers : 4, :],
        ],
        axis=1,
    )

    s0 = tc.Circuit(n_qubits).state()

    def layer_loop(s_in, layer_param_block):
        lambda_l = layer_param_block[0]
        rx_l = layer_param_block[1]
        ry_l = layer_param_block[2]
        rz_l = layer_param_block[3]

        c_ = tc.Circuit(n_qubits, inputs=s_in)

        for q in range(n_qubits):
            c_.rx(q, theta=rx_l[q])
            c_.ry(q, theta=ry_l[q])
            c_.rz(q, theta=rz_l[q])

        for q in range(n_qubits - 1):
            c_.cz(q, q + 1)
        if n_qubits != 2:
            c_.cz(n_qubits - 1, 0)

        for q in range(n_qubits):
            c_.rx(q, theta=X[q] * lambda_l[q])

        return c_.state()

    s_final = K.scan(layer_loop, layer_params, s0)

    c_final = tc.Circuit(n_qubits, inputs=s_final)
    final_offset = 4 * n_layers
    for q in range(n_qubits):
        c_final.rx(q, theta=params[final_offset + 0, q])
        c_final.ry(q, theta=params[final_offset + 1, q])
        c_final.rz(q, theta=params[final_offset + 2, q])

    return c_final


def RX_RZ_RY_RZ_layer_scan(
    X: jax.Array,
    n_qubits: Int,
    params: jax.Array,
    n_layers: Int,
):
    """Predict score from the scan-based reuploading circuit."""
    circuit = build_reuploading_circuit_layer_scan(
        X=X,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
    )
    return jnp.real(circuit.expectation_ps(z=np.arange(int(n_qubits))))


def _prepare_paper_reupload_chunks(
    X: jax.Array,
    n_qubits: int,
    n_layers: int,
    chunk_mode: str = "cyclic",
) -> jax.Array:
    """Prepare x^[l] chunks with shape (n_layers, n_qubits, 3) for paper-style encoding."""
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    required = 3 * n_qubits * n_layers

    x_flat = jnp.asarray(X, dtype=DTYPE).reshape(-1)
    x_dim = int(x_flat.shape[0])
    if x_dim <= 0:
        raise ValueError("Input feature vector must have positive dimension.")

    mode = str(chunk_mode)
    if mode == "strict":
        if x_dim != required:
            raise ValueError(
                f"paper_data_chunk_mode='strict' expects input dim {required}, got {x_dim}."
            )
        packed = x_flat
    elif mode == "cyclic":
        repeats = (required + x_dim - 1) // x_dim
        packed = jnp.tile(x_flat, repeats)[:required]
    else:
        raise ValueError(
            f"Unsupported paper_data_chunk_mode '{mode}'. Expected 'cyclic' or 'strict'."
        )

    return packed.reshape(n_layers, n_qubits, 3)


def build_paper_modular_reuploading_circuit_scan(
    X: jax.Array,
    n_qubits: Int,
    params: jax.Array,
    n_layers: Int,
    n_reuploads: Int = 1,
    chunk_mode: str = "cyclic",
):
    """
    Build paper-style modular reuploading circuit:
      V(x,theta) = prod_{p=1..P} prod_{l=1..L} U_l(theta_{p,l}) R_l(x^[l]).

    Here U_l is implemented as RX/RY/RZ + ring CZ, and R_l uses
    per-qubit RZ(phi_1) -> RY(phi_2) -> RZ(phi_3), where
    x^[l] is chunked to shape (n_qubits, 3).
    """
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    n_reuploads = int(n_reuploads)
    if n_reuploads <= 0:
        raise ValueError(f"n_reuploads must be positive, got {n_reuploads}.")

    params = jnp.asarray(params, dtype=DTYPE)
    expected_shape = (3 * n_reuploads * n_layers, n_qubits)
    if params.shape != expected_shape:
        raise ValueError(f"Expected params shape {expected_shape}, got {tuple(params.shape)}.")

    layer_chunks = _prepare_paper_reupload_chunks(
        X=X,
        n_qubits=n_qubits,
        n_layers=n_layers,
        chunk_mode=chunk_mode,
    )

    total_blocks = n_reuploads * n_layers
    encoding_blocks = jnp.tile(layer_chunks[None, :, :, :], (n_reuploads, 1, 1, 1)).reshape(
        total_blocks, n_qubits, 3
    )
    u_blocks = params.reshape(total_blocks, 3, n_qubits)

    s0 = tc.Circuit(n_qubits).state()

    def block_loop(s_in, block_data):
        theta_block, enc_block = block_data
        c_ = tc.Circuit(n_qubits, inputs=s_in)

        for q in range(n_qubits):
            c_.rx(q, theta=theta_block[0, q])
            c_.ry(q, theta=theta_block[1, q])
            c_.rz(q, theta=theta_block[2, q])

        for q in range(n_qubits - 1):
            c_.cz(q, q + 1)
        if n_qubits != 2:
            c_.cz(n_qubits - 1, 0)

        for q in range(n_qubits):
            phi_1 = enc_block[q, 0]
            phi_2 = enc_block[q, 1]
            phi_3 = enc_block[q, 2]
            c_.rz(q, theta=phi_1)
            c_.ry(q, theta=phi_2)
            c_.rz(q, theta=phi_3)

        return c_.state()

    s_final = K.scan(block_loop, (u_blocks, encoding_blocks), s0)
    return tc.Circuit(n_qubits, inputs=s_final)


def paper_modular_reuploading_predict_scan(
    X: jax.Array,
    n_qubits: Int,
    params: jax.Array,
    n_layers: Int,
    n_reuploads: Int = 1,
    chunk_mode: str = "cyclic",
    observable: str = "z0",
):
    """Predict score from paper-style modular reuploading circuit."""
    circuit = build_paper_modular_reuploading_circuit_scan(
        X=X,
        n_qubits=n_qubits,
        params=params,
        n_layers=n_layers,
        n_reuploads=n_reuploads,
        chunk_mode=chunk_mode,
    )

    obs = str(observable)
    if obs == "z0":
        return jnp.real(circuit.expectation_ps(z=[0]))
    if obs == "global_z":
        return jnp.real(circuit.expectation_ps(z=np.arange(int(n_qubits))))
    raise ValueError(f"Unsupported paper_observable '{obs}'. Expected 'z0' or 'global_z'.")


def initialize_paper_modular_params(
    key: jax.random.PRNGKey,
    n_qubits: int,
    n_layers: int,
    n_reuploads: int,
    scheme: str = "block_normal",
    std: float = 1.0,
    variational_mean: float = 0.0,
    variational_std: float = 0.1,
    teacher_params_mean: jax.Array | None = None,
    teacher_std: float = 0.1,
    clip_abs: float | None = None,
) -> jax.Array:
    """Initialize params for paper-style modular ansatz with shape (3*P*L, n_qubits)."""
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    n_reuploads = int(n_reuploads)
    if n_reuploads <= 0:
        raise ValueError(f"n_reuploads must be positive, got {n_reuploads}.")

    shape = (3 * n_reuploads * n_layers, n_qubits)

    if scheme == "standard_normal":
        params = float(std) * jax.random.normal(key=key, shape=shape, dtype=DTYPE)
    elif scheme == "zeros":
        params = jnp.zeros(shape, dtype=DTYPE)
    elif scheme == "block_normal":
        params = (
            float(variational_mean)
            + float(variational_std) * jax.random.normal(key=key, shape=shape, dtype=DTYPE)
        )
    elif scheme == "teacher_params_normal":
        if teacher_params_mean is None:
            raise ValueError(
                "scheme='teacher_params_normal' requires teacher_params_mean with shape "
                f"{shape}."
            )
        teacher_mean = jnp.asarray(teacher_params_mean, dtype=DTYPE)
        if teacher_mean.shape != shape:
            raise ValueError(
                f"teacher_params_mean shape mismatch: expected {shape}, got {tuple(teacher_mean.shape)}."
            )
        params = teacher_mean + float(teacher_std) * jax.random.normal(key=key, shape=shape, dtype=DTYPE)
    else:
        raise ValueError(
            f"Unsupported param init scheme '{scheme}' for paper_modular ansatz. "
            "Expected one of: 'block_normal', 'standard_normal', 'zeros', 'teacher_params_normal'."
        )

    if clip_abs is not None:
        clip_abs = float(clip_abs)
        if clip_abs > 0.0:
            params = jnp.clip(params, -clip_abs, clip_abs)

    return params


def initialize_reuploading_params(
    key: jax.random.PRNGKey,
    n_qubits: int,
    n_layers: int,
    scheme: str = "block_normal",
    std: float = 1.0,
    encoding_mean: float = 1.0,
    encoding_std: float = 0.1,
    variational_mean: float = 0.0,
    variational_std: float = 0.1,
    final_mean: float = 0.0,
    final_std: float = 0.1,
    teacher_params_mean: jax.Array | None = None,
    teacher_std: float = 0.1,
    clip_abs: float | None = None,
) -> jax.Array:
    """Initialize reuploading-circuit parameters with configurable schemes."""
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)
    shape = (4 * n_layers + 3, n_qubits)

    if scheme == "standard_normal":
        params = float(std) * jax.random.normal(key=key, shape=shape, dtype=DTYPE)
    elif scheme == "zeros":
        params = jnp.zeros(shape, dtype=DTYPE)
    elif scheme == "block_normal":
        key_enc, key_var, key_final = jax.random.split(key, 3)

        params = jnp.zeros(shape, dtype=DTYPE)

        enc = (
            float(encoding_mean)
            + float(encoding_std) * jax.random.normal(key_enc, (n_layers, n_qubits), dtype=DTYPE)
        )
        params = params.at[0 : 4 * n_layers : 4, :].set(enc)

        var = (
            float(variational_mean)
            + float(variational_std)
            * jax.random.normal(key_var, (n_layers, 3, n_qubits), dtype=DTYPE)
        )
        params = params.at[1 : 4 * n_layers : 4, :].set(var[:, 0, :])
        params = params.at[2 : 4 * n_layers : 4, :].set(var[:, 1, :])
        params = params.at[3 : 4 * n_layers : 4, :].set(var[:, 2, :])

        final_block = (
            float(final_mean)
            + float(final_std) * jax.random.normal(key_final, (3, n_qubits), dtype=DTYPE)
        )
        params = params.at[4 * n_layers : 4 * n_layers + 3, :].set(final_block)
    elif scheme == "teacher_params_normal":
        if teacher_params_mean is None:
            raise ValueError(
                "scheme='teacher_params_normal' requires teacher_params_mean with shape "
                f"{shape}."
            )
        teacher_mean = jnp.asarray(teacher_params_mean, dtype=DTYPE)
        if teacher_mean.shape != shape:
            raise ValueError(
                f"teacher_params_mean shape mismatch: expected {shape}, got {tuple(teacher_mean.shape)}."
            )
        params = teacher_mean + float(teacher_std) * jax.random.normal(key=key, shape=shape, dtype=DTYPE)
    else:
        raise ValueError(
            f"Unsupported param init scheme '{scheme}'. "
            "Expected one of: 'block_normal', 'standard_normal', 'zeros', 'teacher_params_normal'."
        )

    if clip_abs is not None:
        clip_abs = float(clip_abs)
        if clip_abs > 0.0:
            params = jnp.clip(params, -clip_abs, clip_abs)

    return params

__all__ = [
    "DTYPE",
    "build_reuploading_circuit_layer_scan",
    "RX_RZ_RY_RZ_layer_scan",
    "build_paper_modular_reuploading_circuit_scan",
    "paper_modular_reuploading_predict_scan",
    "initialize_paper_modular_params",
    "initialize_reuploading_params",
]
