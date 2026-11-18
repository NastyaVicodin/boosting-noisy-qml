"""Dataset helpers."""

try:
    from .loaders import DState, DataLoader, batch_index
except Exception:  # pragma: no cover - Equinox not installed in lightweight test env
    DState = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    batch_index = None  # type: ignore[assignment]

from .datasets import (
    make_toy_binary_classification,
    prepare_binary_fmnist,
    prepare_class_conditional_gaussian_binary_dataset,
    prepare_phase05_binary_data,
    prepare_phase05_binary_data_arrays,
    prepare_phase05_binary_data_split,
    prepare_teacher_student_binary_dataset,
    split_binary_train_validation_arrays,
    train_test_split_binary,
    train_validation_split_binary,
)

__all__ = [
    "DState",
    "DataLoader",
    "batch_index",
    "make_toy_binary_classification",
    "prepare_binary_fmnist",
    "prepare_class_conditional_gaussian_binary_dataset",
    "prepare_phase05_binary_data",
    "prepare_phase05_binary_data_arrays",
    "prepare_phase05_binary_data_split",
    "prepare_teacher_student_binary_dataset",
    "split_binary_train_validation_arrays",
    "train_test_split_binary",
    "train_validation_split_binary",
]
