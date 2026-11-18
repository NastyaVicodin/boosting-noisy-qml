"""Shared EM-style PQC models and teacher-aware initialization helpers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int

from boosting_qml.pqc.ansatz import (
    DTYPE,
    RX_RZ_RY_RZ_layer_scan,
    initialize_paper_modular_params,
    initialize_reuploading_params,
    paper_modular_reuploading_predict_scan,
)
from boosting_qml.pqc.density_matrix import exact_noisy_reuploading_density_score
from boosting_qml.pqc.monte_carlo import (
    sample_noisy_reuploading_score_streaming,
    sample_noisy_reuploading_score_vmap,
)
from boosting_qml.pqc.measurement import (
    sample_pauli_expectation_score,
    validate_integer_shots,
    validate_measurement_gradient_mode,
    validate_measurement_sampling_distribution,
)

def build_teacher_param_mean_for_student_layers(
    teacher_params: jax.Array,
    teacher_n_layers: int,
    student_n_layers: int,
    option: str = "option2",
) -> jax.Array:
    """Build student-shaped teacher-mean params for teacher-based initialization.

    option2:
      - requires teacher_n_layers == student_n_layers
      - returns full teacher parameters as the mean (teacher + std everywhere)

    option1:
      - copy only the overlapping layer blocks [lambda, rx, ry, rz]
      - always copy the teacher final variational block into the student final block
      - all non-overlapping student layer blocks are initialized to zero
    """
    teacher_n_layers = int(teacher_n_layers)
    student_n_layers = int(student_n_layers)

    teacher_params = jnp.asarray(teacher_params, dtype=DTYPE)
    if teacher_params.ndim != 2:
        raise ValueError(
            f"teacher_params must be rank-2, got ndim={teacher_params.ndim}."
        )

    n_qubits = int(teacher_params.shape[1])
    expected_teacher_shape = (4 * teacher_n_layers + 3, n_qubits)
    if teacher_params.shape != expected_teacher_shape:
        raise ValueError(
            "teacher_params shape mismatch: "
            f"expected {expected_teacher_shape}, got {tuple(teacher_params.shape)}."
        )

    student_shape = (4 * student_n_layers + 3, n_qubits)

    if option == "option2":
        if teacher_n_layers != student_n_layers:
            raise ValueError(
                "teacher_param_init_option='option2' requires teacher_n_layers == n_layers, "
                f"got teacher_n_layers={teacher_n_layers}, n_layers={student_n_layers}."
            )
        return teacher_params

    if option == "option1":
        mean_params = jnp.zeros(student_shape, dtype=DTYPE)
        overlap_layers = min(teacher_n_layers, student_n_layers)

        if overlap_layers > 0:
            overlap_rows = 4 * overlap_layers
            mean_params = mean_params.at[:overlap_rows, :].set(teacher_params[:overlap_rows, :])
        mean_params = mean_params.at[4 * student_n_layers : 4 * student_n_layers + 3, :].set(
            teacher_params[4 * teacher_n_layers : 4 * teacher_n_layers + 3, :]
        )

        return mean_params

    raise ValueError(
        f"Unsupported teacher_param_init_option '{option}'. "
        "Expected one of: 'option1', 'option2'."
    )


class ErrorMitigatedPQCModel(eqx.Module):
    params: jax.Array
    n_qubits: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_shots: int = eqx.field(static=True)
    noise_backend: str = eqx.field(static=True)
    n_trajectories: int = eqx.field(static=True)
    trajectory_batch_size: int = eqx.field(static=True)
    noise_strength_1q: float = eqx.field(static=True)
    noise_strength_2q: float = eqx.field(static=True)
    mc_apply_two_qubit_noise: bool = eqx.field(static=True)
    measurement_sampling_distribution: str = eqx.field(static=True)
    measurement_gradient_mode: str = eqx.field(static=True)
    parameter_shift_chunk_size: int = eqx.field(static=True)
    ansatz_kind: str = eqx.field(static=True)
    n_reuploads: int = eqx.field(static=True)
    paper_data_chunk_mode: str = eqx.field(static=True)
    paper_observable: str = eqx.field(static=True)

    def __init__(
        self,
        n_qubits: Int,
        n_layers: Int,
        n_shots: Int,
        key: jax.random.PRNGKey,
        param_init_scheme: str = "block_normal",
        param_init_std: float = 1.0,
        param_init_encoding_mean: float = 1.0,
        param_init_encoding_std: float = 0.1,
        param_init_variational_mean: float = 0.0,
        param_init_variational_std: float = 0.1,
        param_init_final_mean: float = 0.0,
        param_init_final_std: float = 0.1,
        param_init_teacher_mean: jax.Array | None = None,
        param_init_teacher_std: float = 0.1,
        param_init_clip_abs: float | None = None,
        noise_backend: str = "effective_shot",
        n_trajectories: int = 1,
        trajectory_batch_size: int = 16,
        noise_strength_1q: float = 0.0,
        noise_strength_2q: float = 0.0,
        mc_apply_two_qubit_noise: bool = True,
        measurement_sampling_distribution: str = "binomial",
        measurement_gradient_mode: str = "straight_through_expectation",
        parameter_shift_chunk_size: int | str | None = 0,
        ansatz_kind: str = "legacy_reuploading",
        n_reuploads: int = 1,
        paper_data_chunk_mode: str = "cyclic",
        paper_observable: str = "z0",
    ):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_shots = int(n_shots)
        self.noise_backend = str(noise_backend)
        self.n_trajectories = int(n_trajectories)
        self.trajectory_batch_size = int(trajectory_batch_size)
        self.noise_strength_1q = float(noise_strength_1q)
        self.noise_strength_2q = float(noise_strength_2q)
        self.mc_apply_two_qubit_noise = bool(mc_apply_two_qubit_noise)
        self.measurement_sampling_distribution = validate_measurement_sampling_distribution(
            measurement_sampling_distribution
        )
        self.measurement_gradient_mode = validate_measurement_gradient_mode(measurement_gradient_mode)
        if parameter_shift_chunk_size is None or str(parameter_shift_chunk_size).lower() in {"all", "none"}:
            self.parameter_shift_chunk_size = 0
        else:
            self.parameter_shift_chunk_size = int(parameter_shift_chunk_size)
        self.ansatz_kind = str(ansatz_kind)
        self.n_reuploads = int(n_reuploads)
        self.paper_data_chunk_mode = str(paper_data_chunk_mode)
        self.paper_observable = str(paper_observable)

        valid_backends = {"density_matrix", "effective_shot", "mc_vmap", "mc_streaming"}
        if self.noise_backend not in valid_backends:
            raise ValueError(f"noise_backend must be one of {sorted(valid_backends)}, got {self.noise_backend!r}.")
        if self.n_trajectories <= 0:
            raise ValueError(f"n_trajectories must be positive, got {self.n_trajectories}.")
        if self.trajectory_batch_size <= 0:
            raise ValueError(f"trajectory_batch_size must be positive, got {self.trajectory_batch_size}.")
        validate_integer_shots(self.n_shots)
        if self.n_reuploads <= 0:
            raise ValueError(f"n_reuploads must be positive, got {self.n_reuploads}.")

        if self.ansatz_kind == "legacy_reuploading":
            self.params = initialize_reuploading_params(
                key=key,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                scheme=param_init_scheme,
                std=param_init_std,
                encoding_mean=param_init_encoding_mean,
                encoding_std=param_init_encoding_std,
                variational_mean=param_init_variational_mean,
                variational_std=param_init_variational_std,
                final_mean=param_init_final_mean,
                final_std=param_init_final_std,
                teacher_params_mean=param_init_teacher_mean,
                teacher_std=param_init_teacher_std,
                clip_abs=param_init_clip_abs,
            )
        elif self.ansatz_kind == "paper_modular":
            self.params = initialize_paper_modular_params(
                key=key,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_reuploads=self.n_reuploads,
                scheme=param_init_scheme,
                std=param_init_std,
                variational_mean=param_init_variational_mean,
                variational_std=param_init_variational_std,
                teacher_params_mean=param_init_teacher_mean,
                teacher_std=param_init_teacher_std,
                clip_abs=param_init_clip_abs,
            )
        else:
            raise ValueError(
                f"Unsupported ansatz_kind '{self.ansatz_kind}'. "
                "Expected one of: 'legacy_reuploading', 'paper_modular'."
            )

    def get_params(self):
        return self.params

    def add_shot_noise(self, inputs: jax.Array, key_shot_noise: jax.random.PRNGKey):
        if self.noise_backend in {"mc_vmap", "mc_streaming"}:
            if self.ansatz_kind != "legacy_reuploading":
                raise ValueError(
                    "Monte Carlo trajectory backends currently support only "
                    "ansatz_kind='legacy_reuploading'."
                )
            if self.noise_backend == "mc_vmap":
                return sample_noisy_reuploading_score_vmap(
                    key_shot_noise,
                    inputs,
                    n_qubits=self.n_qubits,
                    params=self.params,
                    n_layers=self.n_layers,
                    n_trajectories=self.n_trajectories,
                    noise_strength_1q=self.noise_strength_1q,
                    noise_strength_2q=self.noise_strength_2q,
                    n_shots_eff=self.n_shots,
                    apply_two_qubit_noise=self.mc_apply_two_qubit_noise,
                    observable="global_z",
                    measurement_sampling_distribution=self.measurement_sampling_distribution,
                    measurement_gradient_mode=self.measurement_gradient_mode,
                )
            return sample_noisy_reuploading_score_streaming(
                key_shot_noise,
                inputs,
                n_qubits=self.n_qubits,
                params=self.params,
                n_layers=self.n_layers,
                n_trajectories=self.n_trajectories,
                noise_strength_1q=self.noise_strength_1q,
                noise_strength_2q=self.noise_strength_2q,
                n_shots_eff=self.n_shots,
                trajectory_batch_size=self.trajectory_batch_size,
                apply_two_qubit_noise=self.mc_apply_two_qubit_noise,
                observable="global_z",
                measurement_sampling_distribution=self.measurement_sampling_distribution,
                measurement_gradient_mode=self.measurement_gradient_mode,
            )

        if self.noise_backend == "density_matrix":
            if self.ansatz_kind != "legacy_reuploading":
                raise ValueError(
                    "density_matrix noise_backend currently supports only "
                    "ansatz_kind='legacy_reuploading'."
                )
            ypred = exact_noisy_reuploading_density_score(
                inputs,
                n_qubits=self.n_qubits,
                params=self.params,
                n_layers=self.n_layers,
                noise_strength_1q=self.noise_strength_1q,
                noise_strength_2q=self.noise_strength_2q,
                apply_two_qubit_noise=self.mc_apply_two_qubit_noise,
                observable="global_z",
            )
            return jnp.clip(
                sample_pauli_expectation_score(
                    ypred,
                    key_shot_noise,
                    n_shots=self.n_shots,
                    distribution=self.measurement_sampling_distribution,
                    gradient_mode=self.measurement_gradient_mode,
                ),
                -1.0,
                1.0,
            )

        if self.ansatz_kind == "paper_modular":
            ypred = paper_modular_reuploading_predict_scan(
                X=inputs,
                n_qubits=self.n_qubits,
                params=self.params,
                n_layers=self.n_layers,
                n_reuploads=self.n_reuploads,
                chunk_mode=self.paper_data_chunk_mode,
                observable=self.paper_observable,
            )
        else:
            ypred = RX_RZ_RY_RZ_layer_scan(inputs, self.n_qubits, self.params, self.n_layers)

        return jnp.clip(
            sample_pauli_expectation_score(
                ypred,
                key_shot_noise,
                n_shots=self.n_shots,
                distribution=self.measurement_sampling_distribution,
                gradient_mode=self.measurement_gradient_mode,
            ),
            -1.0,
            1.0,
        )

    def __call__(self, x: jax.Array, key_shot_noise: jax.random.PRNGKey):
        return self.add_shot_noise(x, key_shot_noise)


  

@eqx.filter_jit
def predict(model: ErrorMitigatedPQCModel, batch_inputs: jax.Array, rng_batch: jax.random.PRNGKey):
    rng_batch_noise = jax.random.split(rng_batch, num=batch_inputs.shape[0])
    return eqx.filter_vmap(model, in_axes=(0, 0))(batch_inputs, rng_batch_noise)

__all__ = [
    "build_teacher_param_mean_for_student_layers",
    "ErrorMitigatedPQCModel",
    "predict",
]
