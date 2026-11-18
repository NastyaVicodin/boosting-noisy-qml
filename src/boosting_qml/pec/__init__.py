"""PEC resource accounting and overhead utilities."""

from .accounting import (
    BOOSTING_COMPONENT_MODEL_EVALUATION_FIELDS,
    BOOSTING_COMPONENT_RAW_SHOT_FIELDS,
    ShotAccountingSummary,
    boosting_component_model_evaluations,
    boosting_component_raw_shots,
    build_shot_accounting_summary,
    empty_shot_accounting_dict,
    shot_accounting_to_dict,
    sum_shot_accounting_dicts,
    training_model_evaluations_from_minibatches,
    validate_boosting_component_accounting,
)
from .overhead import (
    pec_sampling_overhead_depolarizing,
    resolve_n_shots_eff,
    resolve_pec_gamma_sq,
    resolve_pec_noise_strengths,
)
from .sampled_training import (
    SAMPLED_PEC_IMPLEMENTATION,
    SampledPECProtocol,
    circuit_pec_gamma,
    estimator_variance,
    sample_signed_pec_mean,
    sampled_pec_parameter_shift_loss_gradient,
    sampled_pec_predict_batch,
    variance_matched_raw_samples,
)

__all__ = [
    "BOOSTING_COMPONENT_MODEL_EVALUATION_FIELDS",
    "BOOSTING_COMPONENT_RAW_SHOT_FIELDS",
    "ShotAccountingSummary",
    "boosting_component_model_evaluations",
    "boosting_component_raw_shots",
    "build_shot_accounting_summary",
    "empty_shot_accounting_dict",
    "pec_sampling_overhead_depolarizing",
    "resolve_n_shots_eff",
    "resolve_pec_gamma_sq",
    "resolve_pec_noise_strengths",
    "shot_accounting_to_dict",
    "sum_shot_accounting_dicts",
    "training_model_evaluations_from_minibatches",
    "validate_boosting_component_accounting",
    "SAMPLED_PEC_IMPLEMENTATION",
    "SampledPECProtocol",
    "circuit_pec_gamma",
    "estimator_variance",
    "sample_signed_pec_mean",
    "sampled_pec_parameter_shift_loss_gradient",
    "sampled_pec_predict_batch",
    "variance_matched_raw_samples",
]
