"""Dataset-size rules for teacher-student PQC sweeps."""

from __future__ import annotations

from math import ceil
from typing import Any


def pqc_parameter_count(n_qubits: int, n_layers: int) -> int:
    """Return the trainable-parameter count for the legacy reuploading PQC."""
    return (4 * int(n_layers) + 3) * int(n_qubits)


def _round_up(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return int(ceil(int(value) / multiple) * multiple)


def _train_size_for_fit_count(fit_train_size: int, validation_fraction: float) -> int:
    """Smallest pre-validation train_size that leaves at least fit_train_size examples."""
    if not (0.0 <= float(validation_fraction) < 1.0):
        raise ValueError("validation_fraction must lie in [0, 1).")
    if validation_fraction <= 0.0:
        return int(fit_train_size)

    train_size = int(ceil(int(fit_train_size) / (1.0 - float(validation_fraction))))
    while train_size - int(round(float(validation_fraction) * train_size)) < int(fit_train_size):
        train_size += 1
    return train_size


def scaled_teacher_student_dataset_sizes(
    *,
    n_qubits: int,
    student_depth: int,
    validation_fraction: float,
    batch_size: int,
    min_fit_train_size: int = 1536,
    samples_per_student_parameter: float = 10.0,
    samples_per_qubit: int = 256,
    min_test_size: int = 1024,
    test_size_ratio: float = 0.5,
    min_teacher_grid_points: int = 10000,
    teacher_grid_multiplier: int = 20,
    scale_depth: int | None = None,
) -> dict[str, Any]:
    """Resolve scaled train/test/grid sizes for a teacher-student dataset.

    The returned ``train_size`` is the pre-validation size. The effective number
    of examples used for fitting is ``train_size - round(validation_fraction *
    train_size)``.
    """
    n_qubits = int(n_qubits)
    student_depth = int(student_depth)
    scale_depth = int(scale_depth) if scale_depth is not None else student_depth
    batch_size = max(1, int(batch_size))

    n_parameters = pqc_parameter_count(n_qubits=n_qubits, n_layers=scale_depth)
    fit_train_size = max(
        int(min_fit_train_size),
        int(ceil(float(samples_per_student_parameter) * n_parameters)),
        int(samples_per_qubit) * n_qubits,
    )
    fit_train_size = _round_up(fit_train_size, batch_size)

    train_size = _train_size_for_fit_count(
        fit_train_size=fit_train_size,
        validation_fraction=float(validation_fraction),
    )
    test_size = max(
        int(min_test_size),
        int(ceil(float(test_size_ratio) * fit_train_size)),
    )
    test_size = _round_up(test_size, batch_size)

    teacher_grid_points = max(
        int(min_teacher_grid_points),
        int(teacher_grid_multiplier) * (int(train_size) + int(test_size)),
    )

    return {
        "mode": "auto",
        "n_qubits": n_qubits,
        "student_depth": student_depth,
        "scale_depth": scale_depth,
        "student_parameter_count_at_scale_depth": n_parameters,
        "min_fit_train_size": int(min_fit_train_size),
        "samples_per_student_parameter": float(samples_per_student_parameter),
        "samples_per_qubit": int(samples_per_qubit),
        "fit_train_size_after_validation": int(fit_train_size),
        "train_size": int(train_size),
        "validation_fraction": float(validation_fraction),
        "test_size": int(test_size),
        "min_test_size": int(min_test_size),
        "test_size_ratio": float(test_size_ratio),
        "teacher_n_grid_points": int(teacher_grid_points),
        "min_teacher_grid_points": int(min_teacher_grid_points),
        "teacher_grid_multiplier": int(teacher_grid_multiplier),
        "batch_size": int(batch_size),
    }
