"""PQC circuit, ansatz, dataset, and simulation helpers."""

from .ansatz import (
    DTYPE,
    RX_RZ_RY_RZ_layer_scan,
    build_paper_modular_reuploading_circuit_scan,
    build_reuploading_circuit_layer_scan,
    initialize_paper_modular_params,
    initialize_reuploading_params,
    paper_modular_reuploading_predict_scan,
)
from .models import (
    ErrorMitigatedPQCModel,
    build_teacher_param_mean_for_student_layers,
    predict,
)
from .monte_carlo import (
    make_reuploading_mc_statuses,
    noisy_reuploading_score_streaming_status,
    noisy_reuploading_score_vmap_status,
    sample_noisy_reuploading_score_streaming,
    sample_noisy_reuploading_score_vmap,
)
from .teacher_student import (
    NoiselessPQCModel,
    Teacher_PQC_lhs,
    load_or_generate_teacher_student_binary_dataset,
    reuploading_circuit,
    teacher_params_from_seed,
    teacher_student_dataset_filename,
)

__all__ = [
    "DTYPE",
    "build_reuploading_circuit_layer_scan",
    "RX_RZ_RY_RZ_layer_scan",
    "build_paper_modular_reuploading_circuit_scan",
    "paper_modular_reuploading_predict_scan",
    "initialize_paper_modular_params",
    "initialize_reuploading_params",
    "build_teacher_param_mean_for_student_layers",
    "ErrorMitigatedPQCModel",
    "predict",
    "NoiselessPQCModel",
    "Teacher_PQC_lhs",
    "load_or_generate_teacher_student_binary_dataset",
    "reuploading_circuit",
    "teacher_params_from_seed",
    "teacher_student_dataset_filename",
    "make_reuploading_mc_statuses",
    "noisy_reuploading_score_streaming_status",
    "noisy_reuploading_score_vmap_status",
    "sample_noisy_reuploading_score_streaming",
    "sample_noisy_reuploading_score_vmap",
]
