"""Validated, atomic persistence for PEC-vs-boosting result artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


VALIDATION_CONFIRMATION_POLICIES = (
    "every_round_independent_stream",
    "monitor_then_confirm_on_success_independent_stream",
)
SCIENTIFIC_IDENTITY_FIELDS = (
    "training_stopping_definition",
    "max_epochs",
    "fixed_min_epochs",
    "early_stopping_patience_epochs",
    "early_stopping_check_frequency_epochs",
    "early_stopping_min_delta",
    "monitored_loss",
    "restore_best_parameters",
    "optimizer",
    "learning_rate",
    "optimizer_beta1",
    "optimizer_beta2",
    "optimizer_epsilon",
    "optimizer_eps_root",
    "batch_size",
    "updates_per_epoch",
    "gradient_clipping",
    "gradient_clip_norm",
    "parameter_initialization_identity",
    "intra_round_target_stopping",
    "target_check_frequency_epochs",
    "check_on_training_stop",
    "candidate_ensemble_construction_identity",
    "edge_estimation_identity",
    "validation_monitor_policy",
    "validation_confirmation_policy",
    "validation_accuracy_tolerance",
    "n_rounds_cap",
    "stop_mode",
    "test_evaluation_policy",
    "target_granularity",
    "target_artifact_sha256",
    "selected_target_value",
    "target_depth",
    "target_task_index",
)


class ResultArtifactValidationError(ValueError):
    """Raised when a PEC-vs-boosting payload is incomplete or inconsistent."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultArtifactValidationError(f"{path} must be a JSON object.")
    return value


def _require_keys(value: Mapping[str, Any], keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ResultArtifactValidationError(f"{path} is missing required keys: {missing}.")


def build_scientific_identity(
    experiment_config: Mapping[str, Any],
    target_identity: Mapping[str, Any] | None,
    weak_learner_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one canonical persisted stopping/target identity."""

    target = {} if target_identity is None else target_identity
    weak = {} if weak_learner_config is None else weak_learner_config
    identity = {
        "training_stopping_definition": weak.get(
            "training_stopping_definition"
        ),
        # Keep max_epochs as the canonical legacy field while making the
        # weak-learner meaning explicit in newly written results.
        "weak_max_epochs": weak.get("n_epochs"),
        "max_epochs": weak.get("n_epochs"),
        "fixed_min_epochs": weak.get("fixed_min_epochs"),
        "early_stopping_patience_epochs": weak.get(
            "early_stopping_patience_epochs"
        ),
        "early_stopping_check_frequency_epochs": weak.get(
            "early_stopping_check_frequency_epochs"
        ),
        "early_stopping_min_delta": weak.get("early_stopping_min_delta"),
        "monitored_loss": weak.get("monitored_loss"),
        "restore_best_parameters": weak.get("restore_best_parameters"),
        "optimizer": weak.get("optimizer", "optax.adam"),
        "learning_rate": weak.get("learning_rate"),
        "optimizer_beta1": weak.get("optimizer_beta1", 0.9),
        "optimizer_beta2": weak.get("optimizer_beta2", 0.999),
        "optimizer_epsilon": weak.get("optimizer_epsilon", 1e-8),
        "optimizer_eps_root": weak.get("optimizer_eps_root", 0.0),
        "batch_size": weak.get("batch_size"),
        "updates_per_epoch": weak.get("updates_per_epoch"),
        "gradient_clipping": weak.get("gradient_clipping", "global_norm"),
        "gradient_clip_norm": weak.get("grad_clip_norm"),
        "parameter_initialization_identity": weak.get("param_init_scheme"),
        "intra_round_target_stopping": weak.get(
            "intra_round_target_stopping"
        ),
        "target_check_frequency_epochs": weak.get(
            "target_check_frequency_epochs"
        ),
        "check_on_training_stop": weak.get("check_on_training_stop"),
        "candidate_ensemble_construction_identity": experiment_config.get(
            "candidate_ensemble_construction_identity"
        ),
        "edge_estimation_identity": experiment_config.get(
            "edge_estimation_identity"
        ),
        "validation_monitor_policy": experiment_config.get(
            "validation_monitor_policy"
        ),
        "validation_confirmation_policy": experiment_config.get(
            "validation_confirmation_policy"
        ),
        "validation_accuracy_tolerance": experiment_config.get(
            "validation_accuracy_tolerance"
        ),
        "n_rounds_cap": experiment_config.get("n_rounds_cap"),
        "stop_mode": experiment_config.get("stop_mode"),
        "test_evaluation_policy": experiment_config.get("test_evaluation_policy"),
        "target_granularity": target.get("target_granularity"),
        "target_artifact_sha256": target.get("target_artifact_sha256"),
        "selected_target_value": target.get(
            "selected_target_value",
            experiment_config.get("fixed_validation_accuracy_target"),
        ),
        "target_depth": target.get("target_depth"),
        "target_task_index": target.get("target_task_index"),
    }
    strict_identity = experiment_config.get("strict_teacher_identity")
    if strict_identity is not None:
        identity["strict_teacher_identity"] = strict_identity
    return identity


def extract_scientific_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the single canonical scientific identity used by write and reuse."""

    run_config = _require_mapping(payload.get("run_config"), "result.run_config")
    identity = _require_mapping(
        run_config.get("scientific_identity"),
        "result.run_config.scientific_identity",
    )
    _require_keys(
        identity,
        SCIENTIFIC_IDENTITY_FIELDS,
        "result.run_config.scientific_identity",
    )
    return {field: identity[field] for field in SCIENTIFIC_IDENTITY_FIELDS}


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return str(left) == str(right)


def validate_pec_comparison_payload(
    payload: Any,
    *,
    require_scientific_identity: bool = True,
) -> dict[str, Any]:
    """Validate the complete public result schema emitted by the comparison CLI."""

    root = _require_mapping(payload, "result")
    _require_keys(
        root,
        (
            "artifact_dir",
            "n_em_runs",
            "margin_threshold",
            "margin_tag",
            "em_run_summaries",
            "margin_summary",
            "validation_summary",
            "stop_target",
            "dataset_stats",
            "algorithm_runs",
            "run_config",
        ),
        "result",
    )
    if not isinstance(root["n_em_runs"], int) or root["n_em_runs"] <= 0:
        raise ResultArtifactValidationError("result.n_em_runs must be a positive integer.")
    em_runs = root["em_run_summaries"]
    if not isinstance(em_runs, list) or len(em_runs) != root["n_em_runs"]:
        raise ResultArtifactValidationError(
            "result.em_run_summaries must contain exactly result.n_em_runs entries."
        )
    _require_mapping(root["dataset_stats"], "result.dataset_stats")
    _require_mapping(root["stop_target"], "result.stop_target")

    run_config = _require_mapping(root["run_config"], "result.run_config")
    _require_keys(
        run_config,
        (
            "em_artifact_config",
            "dataset_config",
            "weak_learner_kind",
            "weak_learner_config",
            "experiment_config",
        ),
        "result.run_config",
    )
    experiment_config = _require_mapping(
        run_config["experiment_config"], "result.run_config.experiment_config"
    )
    experiment_keys = ["algorithms", "rng_seed", "stop_mode", "test_evaluation_policy"]
    if require_scientific_identity:
        experiment_keys.append("validation_confirmation_policy")
    _require_keys(
        experiment_config,
        tuple(experiment_keys),
        "result.run_config.experiment_config",
    )
    policy = experiment_config.get("validation_confirmation_policy")
    if require_scientific_identity and policy not in VALIDATION_CONFIRMATION_POLICIES:
        raise ResultArtifactValidationError(
            "result.run_config.experiment_config.validation_confirmation_policy "
            f"must be one of {list(VALIDATION_CONFIRMATION_POLICIES)}."
        )

    if require_scientific_identity:
        scientific_identity = extract_scientific_identity(root)
        derived_identity = build_scientific_identity(
            experiment_config,
            run_config.get("target_identity")
            if isinstance(run_config.get("target_identity"), Mapping)
            else None,
            run_config["weak_learner_config"],
        )
        inconsistent = [
            field
            for field in SCIENTIFIC_IDENTITY_FIELDS
            if not _same_value(scientific_identity[field], derived_identity[field])
        ]
        if inconsistent:
            raise ResultArtifactValidationError(
                "result.run_config.scientific_identity disagrees with its source "
                f"configuration for fields: {inconsistent}."
            )

    algorithms = root["algorithm_runs"]
    if not isinstance(algorithms, Mapping) or not algorithms:
        raise ResultArtifactValidationError("result.algorithm_runs must be a non-empty object.")
    configured_algorithms = experiment_config["algorithms"]
    if not isinstance(configured_algorithms, (list, tuple)):
        raise ResultArtifactValidationError(
            "result.run_config.experiment_config.algorithms must be an array."
        )
    if set(str(value) for value in configured_algorithms) != set(str(key) for key in algorithms):
        raise ResultArtifactValidationError(
            "result.algorithm_runs does not match the configured algorithms."
        )

    for name, raw_summary in algorithms.items():
        path = f"result.algorithm_runs.{name}"
        summary = _require_mapping(raw_summary, path)
        summary_keys = [
                "algorithm_name",
                "n_rounds_cap",
                "n_rounds_completed",
                "per_round_diagnostics",
                "target_reached",
                "selection_reason",
                "stop_summary",
                "resource_summary",
        ]
        if require_scientific_identity:
            summary_keys.append("validation_confirmation_policy")
        _require_keys(summary, tuple(summary_keys), path)
        if str(summary["algorithm_name"]) != str(name):
            raise ResultArtifactValidationError(
                f"{path}.algorithm_name does not match its object key."
            )
        if (
            require_scientific_identity
            and summary["validation_confirmation_policy"] != policy
        ):
            raise ResultArtifactValidationError(
                f"{path}.validation_confirmation_policy disagrees with the canonical policy."
            )
        completed = summary["n_rounds_completed"]
        if not isinstance(completed, int) or completed <= 0:
            raise ResultArtifactValidationError(
                f"{path}.n_rounds_completed must be a positive integer."
            )
        history = summary["per_round_diagnostics"]
        if not isinstance(history, list) or len(history) != completed:
            raise ResultArtifactValidationError(
                f"{path}.per_round_diagnostics must contain exactly {completed} rounds."
            )
        stop_summary = _require_mapping(summary["stop_summary"], f"{path}.stop_summary")
        _require_keys(
            stop_summary,
            ("target_reached", "stop_round_index", "stop_reason"),
            f"{path}.stop_summary",
        )
        if int(stop_summary["stop_round_index"]) != completed:
            raise ResultArtifactValidationError(
                f"{path}.stop_summary.stop_round_index must equal n_rounds_completed."
            )
        if bool(stop_summary["target_reached"]) != bool(summary["target_reached"]):
            raise ResultArtifactValidationError(
                f"{path}.target_reached disagrees with stop_summary."
            )
        if (
            stop_summary["stop_reason"] == "max_rounds_reached"
            and completed != int(summary["n_rounds_cap"])
        ):
            raise ResultArtifactValidationError(
                f"{path} max-round censoring must end at n_rounds_cap."
            )
        resources = _require_mapping(summary["resource_summary"], f"{path}.resource_summary")
        _require_keys(
            resources,
            (
                "algorithm",
                "n_rounds",
                "shot_accounting",
                "validation_stop_shot_accounting",
                "reported_total_shot_accounting",
            ),
            f"{path}.resource_summary",
        )
        if str(resources["algorithm"]) != str(name) or int(resources["n_rounds"]) != completed:
            raise ResultArtifactValidationError(
                f"{path}.resource_summary algorithm/round count is inconsistent."
            )
        epoch_ledger_keys = {
            "total_weak_learner_epochs",
            "weak_learner_converged_count",
            "weak_learner_censored_count",
        }
        present_epoch_ledger_keys = epoch_ledger_keys.intersection(resources)
        if present_epoch_ledger_keys and present_epoch_ledger_keys != epoch_ledger_keys:
            raise ResultArtifactValidationError(
                f"{path}.resource_summary must persist the complete weak-learner "
                "epoch ledger when any of its fields are present."
            )
        if present_epoch_ledger_keys:
            epoch_sum = sum(
                int(round_row.get("weak_learner_epochs_run") or 0)
                for round_row in history
            )
            converged_count = sum(
                round_row.get("weak_learner_converged") is True
                for round_row in history
            )
            censored_count = sum(
                round_row.get("weak_learner_converged") is False
                for round_row in history
            )
            if int(resources["total_weak_learner_epochs"]) != epoch_sum:
                raise ResultArtifactValidationError(
                    f"{path}.resource_summary.total_weak_learner_epochs does not "
                    "sum the actual per-round epochs."
                )
            if int(resources["weak_learner_converged_count"]) != converged_count:
                raise ResultArtifactValidationError(
                    f"{path}.resource_summary.weak_learner_converged_count is inconsistent."
                )
            if int(resources["weak_learner_censored_count"]) != censored_count:
                raise ResultArtifactValidationError(
                    f"{path}.resource_summary.weak_learner_censored_count is inconsistent."
                )
        if "total_model_evaluations" in resources:
            round_model_evaluations = sum(
                float(round_row.get("total_model_evaluations") or 0.0)
                for round_row in history
            )
            if float(resources["total_model_evaluations"]) != round_model_evaluations:
                raise ResultArtifactValidationError(
                    f"{path}.resource_summary.total_model_evaluations does not "
                    "sum the actual per-round ledger."
                )
    return dict(root)


def atomic_write_pec_comparison_result(path: Path | str, payload: Any) -> Path:
    """Atomically write, parse back, and schema-check a complete result artifact."""

    validated = validate_pec_comparison_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(validated, indent=2, sort_keys=True) + "\n"
    json_normalized = json.loads(rendered)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None

        reparsed = json.loads(destination.read_text(encoding="utf-8"))
        validate_pec_comparison_payload(reparsed)
        if reparsed != json_normalized:
            raise ResultArtifactValidationError(
                f"Persisted result differs from the requested payload: {destination}"
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
