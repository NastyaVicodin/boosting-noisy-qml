"""Shot-accounting helpers for PEC and boosting comparisons.

Most runs use the effective-shot convention: algorithms request an integer
number of shots per expectation value, and PEC raw shots are obtained by
multiplying by the sampling overhead gamma^2.  For physical trajectory sampling,
the trajectory multiplier can be marked as hardware shots so each sampled
channel trajectory contributes to the raw-shot cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


BOOSTING_COMPONENT_RAW_SHOT_FIELDS = (
    "component_train_raw_shots",
    "component_gradient_raw_shots",
    "component_training_monitor_raw_shots",
    "component_edge_raw_shots",
    "component_validation_monitor_raw_shots",
    "component_validation_confirm_raw_shots",
    "component_train_margin_confirm_raw_shots",
    "component_diagnostic_test_raw_shots",
)
BOOSTING_COMPONENT_MODEL_EVALUATION_FIELDS = (
    "component_train_model_evaluations",
    "component_gradient_model_evaluations",
    "component_training_monitor_model_evaluations",
    "component_edge_model_evaluations",
    "component_validation_monitor_model_evaluations",
    "component_validation_confirm_model_evaluations",
    "component_train_margin_confirm_model_evaluations",
    "component_diagnostic_test_model_evaluations",
)


def _component_sum(
    accounting: Mapping[str, Any],
    fields: tuple[str, ...],
) -> float:
    append_only_optional = {
        "component_train_margin_confirm_raw_shots",
        "component_train_margin_confirm_model_evaluations",
    }
    missing = [
        field
        for field in fields
        if accounting.get(field) is None and field not in append_only_optional
    ]
    if missing:
        raise ValueError(f"Boosting component ledger is missing fields: {missing}.")
    values = [float(accounting.get(field, 0) or 0) for field in fields]
    if any(value < 0.0 for value in values):
        raise ValueError("Boosting component ledger contains a negative cost.")
    return float(sum(values))


def boosting_component_raw_shots(accounting: Mapping[str, Any]) -> float:
    """Return the disjoint validation-stop raw-shot component sum."""

    return _component_sum(accounting, BOOSTING_COMPONENT_RAW_SHOT_FIELDS)


def boosting_component_model_evaluations(accounting: Mapping[str, Any]) -> float:
    """Return the disjoint validation-stop model-evaluation component sum."""

    return _component_sum(
        accounting,
        BOOSTING_COMPONENT_MODEL_EVALUATION_FIELDS,
    )


def validate_boosting_component_accounting(
    accounting: Mapping[str, Any],
) -> None:
    """Require aggregate totals to equal their disjoint component ledgers."""

    raw_components = boosting_component_raw_shots(accounting)
    model_components = boosting_component_model_evaluations(accounting)
    raw_total = float(accounting.get("total_raw_shots", 0.0) or 0.0)
    model_total = float(accounting.get("total_model_evaluations", 0.0) or 0.0)
    if abs(raw_components - raw_total) > max(1e-9, abs(raw_total) * 1e-12):
        raise ValueError(
            "Boosting raw-shot accounting is non-additive: "
            f"components={raw_components}, total_raw_shots={raw_total}."
        )
    if abs(model_components - model_total) > max(1e-9, abs(model_total) * 1e-12):
        raise ValueError(
            "Boosting model-evaluation accounting is non-additive: "
            f"components={model_components}, total_model_evaluations={model_total}."
        )


@dataclass(frozen=True)
class ShotAccountingSummary:
    """Train/inference shot accounting for one run or one weak learner."""

    n_shots_eff_per_expectation: int
    raw_shot_overhead: float = 1.0
    train_model_evaluations: int = 0
    inference_model_evaluations: int = 0
    trajectory_multiplier: int = 1
    trajectory_multiplier_counts_as_shots: bool = False
    source: str = "unknown"

    @property
    def total_model_evaluations(self) -> int:
        return int(self.train_model_evaluations + self.inference_model_evaluations)

    @property
    def train_effective_shots(self) -> float:
        return float(
            self.train_model_evaluations
            * self.n_shots_eff_per_expectation
            * self._shot_trajectory_multiplier
        )

    @property
    def inference_effective_shots(self) -> float:
        return float(
            self.inference_model_evaluations
            * self.n_shots_eff_per_expectation
            * self._shot_trajectory_multiplier
        )

    @property
    def total_effective_shots(self) -> float:
        return float(self.train_effective_shots + self.inference_effective_shots)

    @property
    def train_raw_shots(self) -> float:
        return float(self.train_effective_shots * self.raw_shot_overhead)

    @property
    def inference_raw_shots(self) -> float:
        return float(self.inference_effective_shots * self.raw_shot_overhead)

    @property
    def total_raw_shots(self) -> float:
        return float(self.train_raw_shots + self.inference_raw_shots)

    @property
    def train_trajectory_evaluations(self) -> int:
        return int(self.train_model_evaluations * self.trajectory_multiplier)

    @property
    def inference_trajectory_evaluations(self) -> int:
        return int(self.inference_model_evaluations * self.trajectory_multiplier)

    @property
    def total_trajectory_evaluations(self) -> int:
        return int(self.train_trajectory_evaluations + self.inference_trajectory_evaluations)

    @property
    def _shot_trajectory_multiplier(self) -> int:
        return int(self.trajectory_multiplier if self.trajectory_multiplier_counts_as_shots else 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "total_model_evaluations": self.total_model_evaluations,
                "train_effective_shots": self.train_effective_shots,
                "inference_effective_shots": self.inference_effective_shots,
                "total_effective_shots": self.total_effective_shots,
                "train_raw_shots": self.train_raw_shots,
                "inference_raw_shots": self.inference_raw_shots,
                "total_raw_shots": self.total_raw_shots,
                "train_trajectory_evaluations": self.train_trajectory_evaluations,
                "inference_trajectory_evaluations": self.inference_trajectory_evaluations,
                "total_trajectory_evaluations": self.total_trajectory_evaluations,
            }
        )
        return payload


def build_shot_accounting_summary(
    *,
    n_shots_eff_per_expectation: int,
    train_model_evaluations: int = 0,
    inference_model_evaluations: int = 0,
    raw_shot_overhead: float = 1.0,
    trajectory_multiplier: int = 1,
    trajectory_multiplier_counts_as_shots: bool = False,
    source: str = "unknown",
) -> ShotAccountingSummary:
    """Build validated shot accounting for one component."""

    n_shots_eff = int(n_shots_eff_per_expectation)
    train_evals = int(train_model_evaluations)
    inference_evals = int(inference_model_evaluations)
    overhead = float(raw_shot_overhead)
    trajectory_mult = int(trajectory_multiplier)

    if n_shots_eff <= 0:
        raise ValueError(f"n_shots_eff_per_expectation must be positive, got {n_shots_eff}.")
    if train_evals < 0 or inference_evals < 0:
        raise ValueError("Model-evaluation counts must be nonnegative.")
    if overhead < 1.0:
        raise ValueError(f"raw_shot_overhead must be >= 1, got {overhead}.")
    if trajectory_mult <= 0:
        raise ValueError(f"trajectory_multiplier must be positive, got {trajectory_mult}.")

    return ShotAccountingSummary(
        n_shots_eff_per_expectation=n_shots_eff,
        raw_shot_overhead=overhead,
        train_model_evaluations=train_evals,
        inference_model_evaluations=inference_evals,
        trajectory_multiplier=trajectory_mult,
        trajectory_multiplier_counts_as_shots=bool(trajectory_multiplier_counts_as_shots),
        source=str(source),
    )


def empty_shot_accounting_dict(source: str = "none") -> dict[str, Any]:
    """Return a zero-cost accounting payload for non-quantum learners."""

    return {
        "source": str(source),
        "n_shots_eff_per_expectation": 0,
        "raw_shot_overhead": 1.0,
        "trajectory_multiplier": 1,
        "trajectory_multiplier_counts_as_shots": False,
        "configured_n_shots_eff": 0,
        "shots_per_expectation": 0,
        "measurement_sampling_distribution": None,
        "measurement_gradient_mode": None,
        "gradient_loss_estimator": None,
        "gradient_unshifted_shots": 0.0,
        "gradient_shifted_shots": 0.0,
        "shifted_parameter_count": 0,
        "shifted_gate_occurrence_count": 0,
        "unshifted_loss_evaluations": 0,
        "shifted_gradient_evaluations": 0,
        "training_monitor_evaluations": 0,
        "optimizer_updates": 0,
        "gradient_evaluations": 0,
        "component_train_model_evaluations": 0,
        "component_gradient_model_evaluations": 0,
        "component_training_monitor_model_evaluations": 0,
        "component_edge_model_evaluations": 0,
        "component_validation_monitor_model_evaluations": 0,
        "component_validation_confirm_model_evaluations": 0,
        "component_train_margin_confirm_model_evaluations": 0,
        "component_diagnostic_test_model_evaluations": 0,
        "component_train_raw_shots": 0.0,
        "component_gradient_raw_shots": 0.0,
        "component_training_monitor_raw_shots": 0.0,
        "component_edge_raw_shots": 0.0,
        "component_validation_monitor_raw_shots": 0.0,
        "component_validation_confirm_raw_shots": 0.0,
        "component_train_margin_confirm_raw_shots": 0.0,
        "component_diagnostic_test_raw_shots": 0.0,
        "boosting_component_ledger_basis": None,
        "mc_status_mode": None,
        "mc_refresh_granularity": None,
        "mc_trajectories_per_expectation": 0,
        "physical_noise_coupling": None,
        "parameter_shift_noise_coupling": None,
        "double_stochastic_simulation": False,
        "train_model_evaluations": 0,
        "inference_model_evaluations": 0,
        "total_model_evaluations": 0,
        "train_effective_shots": 0.0,
        "inference_effective_shots": 0.0,
        "total_effective_shots": 0.0,
        "train_raw_shots": 0.0,
        "inference_raw_shots": 0.0,
        "total_raw_shots": 0.0,
        "train_trajectory_evaluations": 0,
        "inference_trajectory_evaluations": 0,
        "total_trajectory_evaluations": 0,
    }


def shot_accounting_to_dict(value: ShotAccountingSummary | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize accounting payloads to a plain JSON-serializable dict."""

    if value is None:
        return empty_shot_accounting_dict()
    if isinstance(value, ShotAccountingSummary):
        return value.to_dict()
    payload = empty_shot_accounting_dict(source=str(value.get("source", "unknown")))
    for key, item in value.items():
        if key in payload:
            payload[key] = item
    return payload


def sum_shot_accounting_dicts(values: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Aggregate per-round shot accounting dictionaries."""

    total = empty_shot_accounting_dict(source="aggregate")
    max_n_shots_eff = 0
    max_overhead = 1.0
    max_trajectory_multiplier = 1
    max_configured_n_shots_eff = 0
    max_shots_per_expectation = 0
    max_mc_trajectories_per_expectation = 0
    max_shifted_parameter_count = 0
    max_shifted_gate_occurrence_count = 0
    trajectory_multiplier_counts_as_shots = False
    double_stochastic_simulation = False
    string_metadata: dict[str, set[str]] = {
        "measurement_sampling_distribution": set(),
        "measurement_gradient_mode": set(),
        "gradient_loss_estimator": set(),
        "boosting_component_ledger_basis": set(),
        "mc_status_mode": set(),
        "mc_refresh_granularity": set(),
        "physical_noise_coupling": set(),
        "parameter_shift_noise_coupling": set(),
    }

    for value in values:
        payload = shot_accounting_to_dict(value)
        max_n_shots_eff = max(max_n_shots_eff, int(payload.get("n_shots_eff_per_expectation", 0) or 0))
        max_overhead = max(max_overhead, float(payload.get("raw_shot_overhead", 1.0) or 1.0))
        max_trajectory_multiplier = max(
            max_trajectory_multiplier,
            int(payload.get("trajectory_multiplier", 1) or 1),
        )
        trajectory_multiplier_counts_as_shots = bool(
            trajectory_multiplier_counts_as_shots
            or payload.get("trajectory_multiplier_counts_as_shots", False)
        )
        double_stochastic_simulation = bool(
            double_stochastic_simulation or payload.get("double_stochastic_simulation", False)
        )
        max_configured_n_shots_eff = max(
            max_configured_n_shots_eff,
            int(payload.get("configured_n_shots_eff", 0) or 0),
        )
        max_shots_per_expectation = max(
            max_shots_per_expectation,
            int(payload.get("shots_per_expectation", 0) or 0),
        )
        max_mc_trajectories_per_expectation = max(
            max_mc_trajectories_per_expectation,
            int(payload.get("mc_trajectories_per_expectation", 0) or 0),
        )
        max_shifted_parameter_count = max(
            max_shifted_parameter_count,
            int(payload.get("shifted_parameter_count", 0) or 0),
        )
        max_shifted_gate_occurrence_count = max(
            max_shifted_gate_occurrence_count,
            int(payload.get("shifted_gate_occurrence_count", 0) or 0),
        )
        for key, values in string_metadata.items():
            item = payload.get(key)
            if item is not None:
                values.add(str(item))
        for key in (
            "train_model_evaluations",
            "inference_model_evaluations",
            "total_model_evaluations",
            "train_effective_shots",
            "inference_effective_shots",
            "total_effective_shots",
            "train_raw_shots",
            "inference_raw_shots",
            "total_raw_shots",
            "train_trajectory_evaluations",
            "inference_trajectory_evaluations",
            "total_trajectory_evaluations",
            "gradient_unshifted_shots",
            "gradient_shifted_shots",
            "unshifted_loss_evaluations",
            "shifted_gradient_evaluations",
            "training_monitor_evaluations",
            "optimizer_updates",
            "gradient_evaluations",
            "component_train_model_evaluations",
            "component_gradient_model_evaluations",
            "component_training_monitor_model_evaluations",
            "component_edge_model_evaluations",
            "component_validation_monitor_model_evaluations",
            "component_validation_confirm_model_evaluations",
            "component_train_margin_confirm_model_evaluations",
            "component_diagnostic_test_model_evaluations",
            "component_train_raw_shots",
            "component_gradient_raw_shots",
            "component_training_monitor_raw_shots",
            "component_edge_raw_shots",
            "component_validation_monitor_raw_shots",
            "component_validation_confirm_raw_shots",
            "component_train_margin_confirm_raw_shots",
            "component_diagnostic_test_raw_shots",
        ):
            total[key] += payload.get(key, 0) or 0

    total["n_shots_eff_per_expectation"] = max_n_shots_eff
    total["raw_shot_overhead"] = max_overhead
    total["trajectory_multiplier"] = max_trajectory_multiplier
    total["trajectory_multiplier_counts_as_shots"] = trajectory_multiplier_counts_as_shots
    total["configured_n_shots_eff"] = max_configured_n_shots_eff
    total["shots_per_expectation"] = max_shots_per_expectation
    total["mc_trajectories_per_expectation"] = max_mc_trajectories_per_expectation
    total["shifted_parameter_count"] = max_shifted_parameter_count
    total["shifted_gate_occurrence_count"] = max_shifted_gate_occurrence_count
    total["double_stochastic_simulation"] = double_stochastic_simulation
    for key, values in string_metadata.items():
        if not values:
            total[key] = None
        elif len(values) == 1:
            total[key] = next(iter(values))
        else:
            total[key] = sorted(values)
    return total


def training_model_evaluations_from_minibatches(
    *,
    n_samples: int,
    batch_size: int,
    n_epochs: int,
    drop_last: bool = True,
) -> int:
    """Count per-example model evaluations made by minibatch training."""

    n_samples = int(n_samples)
    batch_size = int(batch_size)
    n_epochs = int(n_epochs)
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_epochs < 0:
        raise ValueError("n_epochs must be nonnegative.")

    if drop_last:
        examples_per_epoch = (n_samples // batch_size) * batch_size
    else:
        examples_per_epoch = n_samples
    return int(examples_per_epoch * n_epochs)
