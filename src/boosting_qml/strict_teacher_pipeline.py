"""Versioned strict teacher-task selection and final 5x5 experiment design.

This module contains the scientific, side-effect-free core of the staged
workflow.  Numerical candidate evaluation and reference optimisation are
deliberately separate stages; selection consumes only their frozen records.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRICT_POLICY = "strict_signal_then_reference_learnability.first5.v1"
STRICT_REGISTRY_SCHEMA = "teacher_task_registry.strict_signal_reference_first5.v1"
STRICT_TARGET_SCHEMA = "pec_boosting.depth_task_targets.strict_teachers_5x5.v1"
CANDIDATE_SEED_SCHEMA = "teacher_candidate_seed_list.depths1_5.v1"
CANDIDATE_SEED_SCHEMA_V2 = "teacher_candidate_seed_list.depths1_5.v2"
CANDIDATE_EXTENSION_POLICY = "append_only_preserve_parent_prefix.v1"
ACCOUNTING_SCHEMA = "component_disjoint_additive_shot_ledger.v2"
DEPTHS = (1, 2, 3, 4, 5)
TASK_INDICES = (0, 1, 2, 3, 4)
CALIBRATION_INDICES = (0, 1, 2, 3, 4)
HELDOUT_INDICES = (5, 6, 7, 8, 9)
BOOSTING_INDICES = (0, 1, 2, 3, 4)
LEARNABILITY_FLOORS = {1: 0.90, 2: 0.90, 3: 0.80, 4: 0.80, 5: 0.75}
SIGNAL_THRESHOLDS = {
    "min_probability_std": 0.09,
    "min_positive_fraction": 0.10,
    "min_negative_fraction": 0.10,
    "max_margin_rejection_fraction": 0.75,
    "q05_strict_upper_bound": 0.4,
    "q95_strict_lower_bound": 0.6,
}
TRAINING_STOPPING_IDENTITY = {
    "training_stopping_definition": "best_loss_no_improvement_patience.v1",
    "max_epochs": 500,
    "fixed_min_epochs": None,
    "early_stopping_patience_epochs": 20,
    "early_stopping_check_frequency_epochs": 1,
    "early_stopping_min_delta": 0.0,
    "restore_best_parameters": True,
}


class StrictTeacherPipelineError(ValueError):
    """Raised when strict selection or its frozen artifacts are invalid."""


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_probability_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise StrictTeacherPipelineError("Teacher probabilities must be a non-empty vector.")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise StrictTeacherPipelineError("Teacher probabilities must be finite values in [0, 1].")
    return array


def strict_signal_summary(
    values: Sequence[float],
    *,
    required_unique_candidates_per_class: int,
    teacher_margin: float = 0.2,
    thresholds: Mapping[str, float] = SIGNAL_THRESHOLDS,
) -> dict[str, Any]:
    """Apply the established legacy signal definitions without changing edges.

    Fractions and population standard deviation use the complete candidate
    grid.  With margin 0.2, negative is ``p < 0.4``, positive is ``p >= 0.6``,
    and the remaining points are rejected.  NumPy's established quantile
    definition is retained.
    """

    array = _finite_probability_values(values)
    lower = 0.5 - float(teacher_margin) / 2.0
    upper = 0.5 + float(teacher_margin) / 2.0
    negative = array < lower
    positive = array >= upper
    rejected = ~(negative | positive)
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    negative_count = int(np.sum(negative))
    positive_count = int(np.sum(positive))
    metrics = {
        "prob_mean": float(np.mean(array)),
        "prob_std": float(np.std(array)),
        "prob_variance": float(np.var(array)),
        "prob_min": float(np.min(array)),
        "prob_q01": float(quantiles[0]),
        "prob_q05": float(quantiles[1]),
        "prob_q25": float(quantiles[2]),
        "prob_median": float(quantiles[3]),
        "prob_q75": float(quantiles[4]),
        "prob_q95": float(quantiles[5]),
        "prob_q99": float(quantiles[6]),
        "prob_max": float(np.max(array)),
        "negative_count": negative_count,
        "positive_count": positive_count,
        "rejected_margin_count": int(np.sum(rejected)),
        "negative_fraction": float(np.mean(negative)),
        "positive_fraction": float(np.mean(positive)),
        "rejected_margin_fraction": float(np.mean(rejected)),
        "required_unique_candidates_per_class": int(
            required_unique_candidates_per_class
        ),
        "teacher_margin": float(teacher_margin),
        "negative_threshold_definition": "p < 0.5 - teacher_margin/2",
        "positive_threshold_definition": "p >= 0.5 + teacher_margin/2",
        "fraction_denominator": "complete_latin_hypercube_candidate_grid",
        "std_definition": "numpy_population_std_ddof0",
        "quantile_definition": "numpy_quantile_default_linear",
    }
    checks = {
        "check_probability_std": metrics["prob_std"]
        >= float(thresholds["min_probability_std"]),
        "check_positive_fraction": metrics["positive_fraction"]
        >= float(thresholds["min_positive_fraction"]),
        "check_negative_fraction": metrics["negative_fraction"]
        >= float(thresholds["min_negative_fraction"]),
        "check_margin_rejection_fraction": metrics["rejected_margin_fraction"]
        <= float(thresholds["max_margin_rejection_fraction"]),
        "check_q05_crosses_negative_threshold": metrics["prob_q05"]
        < float(thresholds["q05_strict_upper_bound"]),
        "check_q95_crosses_positive_threshold": metrics["prob_q95"]
        > float(thresholds["q95_strict_lower_bound"]),
        "check_unique_candidates": min(negative_count, positive_count)
        >= int(required_unique_candidates_per_class),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **metrics,
        "screening_policy": STRICT_POLICY,
        "signal_thresholds": {key: float(value) for key, value in thresholds.items()},
        **checks,
        "signal_accepted": not failed,
        "signal_failure_reasons": failed,
    }


def calibration_median(accuracies: Sequence[float]) -> float:
    if len(accuracies) != 5:
        raise StrictTeacherPipelineError("Exactly five calibration accuracies are required.")
    values = [float(value) for value in accuracies]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise StrictTeacherPipelineError("Calibration accuracies must be finite values in [0, 1].")
    return float(statistics.median(values))


def candidate_decision(record: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
    """Return the two-stage admission decision for one frozen candidate."""

    signal = record.get("signal_metrics")
    if not isinstance(signal, Mapping):
        raise StrictTeacherPipelineError("Candidate is missing signal_metrics.")
    floor = float(LEARNABILITY_FLOORS[int(depth)])
    signal_pass = bool(signal.get("signal_accepted"))
    accuracies = record.get("calibration_validation_accuracies")
    if isinstance(accuracies, Sequence) and not isinstance(
        accuracies, (str, bytes)
    ) and len(accuracies) == 5:
        median = calibration_median(accuracies)
        learnability_pass = median >= floor
    elif signal_pass:
        if not isinstance(accuracies, Sequence) or isinstance(accuracies, (str, bytes)):
            raise StrictTeacherPipelineError(
                "Signal-valid candidate is missing calibration_validation_accuracies."
            )
        raise StrictTeacherPipelineError(
            "Signal-valid candidate does not have exactly five calibration accuracies."
        )
    else:
        if accuracies is not None and accuracies != () and accuracies != []:
            raise StrictTeacherPipelineError(
                "Signal-invalid candidate has an incomplete calibration record."
            )
        median: float | None = None
        learnability_pass: bool | None = None
    reasons = []
    if not signal_pass:
        reasons.extend(str(value) for value in signal.get("signal_failure_reasons", []))
        if not reasons:
            reasons.append("strict_signal_failed")
    if signal_pass and not learnability_pass:
        reasons.append("median_reference_validation_accuracy_below_depth_floor")
    return {
        "signal_pass": signal_pass,
        "reference_learnability_pass": learnability_pass,
        "median_reference_validation_accuracy": median,
        "reference_learnability_floor": floor,
        "accepted": signal_pass and bool(learnability_pass),
        "failure_reasons": reasons,
    }


def select_first_passing_candidates(
    candidates_by_depth: Mapping[int | str, Sequence[Mapping[str, Any]]],
    *,
    accepted_per_depth: int = 5,
) -> dict[int, list[dict[str, Any]]]:
    """Select in declared order, never by accuracy rank."""

    selected: dict[int, list[dict[str, Any]]] = {}
    for depth in DEPTHS:
        records = list(
            candidates_by_depth.get(depth, candidates_by_depth.get(str(depth), []))
        )
        candidate_indices = [int(row["candidate_index"]) for row in records]
        if candidate_indices != sorted(candidate_indices) or len(candidate_indices) != len(
            set(candidate_indices)
        ):
            raise StrictTeacherPipelineError(
                f"Depth {depth} candidates are not in unique declared candidate order."
            )
        accepted: list[dict[str, Any]] = []
        for raw in records:
            record = dict(raw)
            decision = candidate_decision(record, depth=depth)
            record.update(decision)
            record["depth"] = depth
            record["acceptance_order"] = (
                len(accepted) if decision["accepted"] and len(accepted) < accepted_per_depth else None
            )
            if decision["accepted"] and len(accepted) < accepted_per_depth:
                record["task_index"] = len(accepted)
                accepted.append(record)
        if len(accepted) != accepted_per_depth:
            raise StrictTeacherPipelineError(
                f"Declared candidate list yielded {len(accepted)} accepted teachers "
                f"at depth {depth}; exactly {accepted_per_depth} are required. "
                "Create an explicit versioned candidate-list extension."
            )
        selected[depth] = accepted
    return selected


def load_candidate_seed_artifact(path: Path | str) -> tuple[dict[str, Any], str]:
    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema not in {CANDIDATE_SEED_SCHEMA, CANDIDATE_SEED_SCHEMA_V2}:
        raise StrictTeacherPipelineError("Candidate seed artifact schema is incompatible.")
    by_depth = payload.get("candidate_seeds_by_depth")
    if not isinstance(by_depth, Mapping) or set(by_depth) != {str(d) for d in DEPTHS}:
        raise StrictTeacherPipelineError("Candidate seed artifact must cover depths 1--5.")
    all_seeds: set[int] = set()
    for depth in DEPTHS:
        seeds = by_depth[str(depth)]
        if not isinstance(seeds, list) or not seeds:
            raise StrictTeacherPipelineError(f"Depth {depth} candidate seed list is empty.")
        numeric = [int(value) for value in seeds]
        if len(numeric) != len(set(numeric)):
            raise StrictTeacherPipelineError(f"Depth {depth} has duplicate candidate seeds.")
        if schema == CANDIDATE_SEED_SCHEMA and all_seeds.intersection(numeric):
            raise StrictTeacherPipelineError("Candidate seeds must be distinct across depths.")
        all_seeds.update(numeric)
    if schema == CANDIDATE_SEED_SCHEMA_V2:
        if payload.get("candidate_list_version") != "v2":
            raise StrictTeacherPipelineError("Candidate v2 version marker is invalid.")
        if payload.get("extension_policy") != CANDIDATE_EXTENSION_POLICY:
            raise StrictTeacherPipelineError("Candidate v2 extension policy is invalid.")
        parent_value = payload.get("parent_artifact_path")
        parent_sha = payload.get("parent_artifact_sha256")
        if not isinstance(parent_value, str) or not parent_value:
            raise StrictTeacherPipelineError("Candidate v2 parent artifact path is missing.")
        parent_path = Path(parent_value)
        if not parent_path.is_absolute():
            parent_path = PROJECT_ROOT / parent_path
        parent, observed_parent_sha = load_candidate_seed_artifact(parent_path)
        if parent.get("schema_version") != CANDIDATE_SEED_SCHEMA:
            raise StrictTeacherPipelineError("Candidate v2 parent must be the v1 artifact.")
        if parent_sha != observed_parent_sha:
            raise StrictTeacherPipelineError("Candidate v2 parent artifact SHA mismatch.")
        parent_by_depth = parent["candidate_seeds_by_depth"]
        for depth in DEPTHS:
            old = [int(value) for value in parent_by_depth[str(depth)]]
            new = [int(value) for value in by_depth[str(depth)]]
            if new[: len(old)] != old:
                raise StrictTeacherPipelineError(
                    f"Candidate v2 depth {depth} changed its v1 prefix."
                )
            expected_count = 20 if depth <= 3 else 120
            if len(new) != expected_count:
                raise StrictTeacherPipelineError(
                    f"Candidate v2 depth {depth} must contain {expected_count} entries."
                )
            if depth <= 3 and new != old:
                raise StrictTeacherPipelineError(
                    f"Candidate v2 must not extend depth {depth}."
                )
            expected_seeds = [
                100_000 + 100 * depth + candidate_index
                for candidate_index in range(expected_count)
            ]
            if new != expected_seeds:
                raise StrictTeacherPipelineError(
                    f"Candidate v2 depth {depth} violates deterministic seed order."
                )
        entries_sha = canonical_sha256(by_depth)
        if payload.get("v2_candidate_entries_sha256") != entries_sha:
            raise StrictTeacherPipelineError(
                "Candidate v2 entries SHA does not match its declared seed lists."
            )
    return dict(payload), file_sha256(candidate_path)


def candidate_seed_parent_identity(
    path: Path | str,
) -> tuple[dict[str, Any], str, Path] | None:
    """Return the validated direct parent of an append-only candidate list."""

    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path
    payload, _ = load_candidate_seed_artifact(candidate_path)
    if payload.get("schema_version") != CANDIDATE_SEED_SCHEMA_V2:
        return None
    parent_path = Path(str(payload["parent_artifact_path"]))
    if not parent_path.is_absolute():
        parent_path = PROJECT_ROOT / parent_path
    parent, parent_sha = load_candidate_seed_artifact(parent_path)
    return parent, parent_sha, parent_path


def strict_identity_from_config(
    config: Mapping[str, Any],
    *,
    candidate_seed_list_sha256: str,
    accepted_registry_sha256: str | None = None,
    target_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    grid = dict(config.get("grid_documentation") or config.get("grid") or {})
    training = dict(config.get("training") or {})
    protocol = dict(config.get("scientific_protocol") or {})
    selection = dict(config.get("strict_teacher_selection") or {})
    configured_thresholds = selection.get("signal_thresholds")
    configured_floors = selection.get("learnability_floors")
    if configured_thresholds != SIGNAL_THRESHOLDS:
        raise StrictTeacherPipelineError(
            "Config strict signal thresholds differ from the frozen v1 policy."
        )
    if configured_floors != {
        str(key): value for key, value in LEARNABILITY_FLOORS.items()
    }:
        raise StrictTeacherPipelineError(
            "Config reference-learnability floors differ from the frozen v1 policy."
        )
    if int(selection.get("accepted_teachers_per_depth", -1)) != 5:
        raise StrictTeacherPipelineError(
            "Config must require exactly five accepted teachers per depth."
        )
    if selection.get("selection_rule") != (
        "first_five_passing_in_declared_candidate_order"
    ):
        raise StrictTeacherPipelineError("Config strict selection rule is incompatible.")
    configured_stopping = {
        "training_stopping_definition": training.get(
            "training_stopping_definition"
        ),
        "max_epochs": training.get("em_max_epochs"),
        "fixed_min_epochs": training.get("fixed_min_epochs"),
        "early_stopping_patience_epochs": training.get(
            "early_stopping_patience_epochs"
        ),
        "early_stopping_check_frequency_epochs": training.get(
            "early_stopping_check_frequency_epochs"
        ),
        "early_stopping_min_delta": training.get("early_stopping_min_delta"),
        "restore_best_parameters": training.get("restore_best_parameters"),
    }
    if configured_stopping != TRAINING_STOPPING_IDENTITY:
        raise StrictTeacherPipelineError(
            "Config training stopping differs from the frozen symmetric protocol."
        )
    if not bool(training.get("early_stopping_enabled", False)):
        raise StrictTeacherPipelineError("Adaptive stopping must remain enabled.")
    if int(training.get("weak_max_epochs", -1)) != 500:
        raise StrictTeacherPipelineError("Weak-learner epoch cap must remain 500.")
    return {
        "teacher_screening_policy": STRICT_POLICY,
        "signal_thresholds": dict(SIGNAL_THRESHOLDS),
        "reference_learnability_floors": {
            str(key): value for key, value in LEARNABILITY_FLOORS.items()
        },
        "candidate_seed_list_sha256": str(candidate_seed_list_sha256),
        "accepted_registry_sha256": accepted_registry_sha256,
        "target_artifact_sha256": target_artifact_sha256,
        "teachers_per_depth": 5,
        "calibration_run_indices": list(CALIBRATION_INDICES),
        "heldout_run_indices": list(HELDOUT_INDICES),
        "boosting_run_indices": list(BOOSTING_INDICES),
        "calibration_optimizer_seed_map": {
            str(run): {
                "train_key": 1000 + run,
                "test_key": 2000 + run,
                "validation_key": 3000 + run,
            }
            for run in CALIBRATION_INDICES
        },
        "heldout_optimizer_seed_map": {
            str(run): {
                "train_key": 1000 + run,
                "test_key": 2000 + run,
                "validation_key": 3000 + run,
            }
            for run in HELDOUT_INDICES
        },
        "boosting_seed_derivation": (
            "boosting_seed_base + depth*1000 + task_index*10 + run_index"
        ),
        "boosting_seed_base": int(config.get("boosting_seed_base", 410_000)),
        "boosting_seed_offsets_per_task": list(BOOSTING_INDICES),
        "noise_levels": [float(value) for value in grid.get("noise_levels", [])],
        "optimizer": "optax.adam",
        "learning_rate": float(training.get("learning_rate", 0.01)),
        "batch_size": int(training.get("batch_size", grid.get("batch_size", 64))),
        "gradient_clipping": "global_norm",
        "gradient_clip_norm": float(training.get("grad_clip_norm", 1.0)),
        "parameter_initialization_identity": str(
            protocol.get("student_initialization", "block_normal")
        ),
        "training_stopping_identity": dict(TRAINING_STOPPING_IDENTITY),
        "physical_noise_identity": {
            "backend": protocol.get("noise_backend", "density_matrix"),
            "channel": "exact_density_matrix_depolarizing",
            "p1_equals_p2": float(config.get("noise_strength_2q_scale", 1.0)) == 1.0,
            "measurement_sampling_distribution": protocol.get(
                "measurement_sampling_distribution", "binomial"
            ),
            "uniform_attenuation": False,
        },
        "accounting_schema_identity": ACCOUNTING_SCHEMA,
    }


def _calibration_sources(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = record.get("calibration_sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise StrictTeacherPipelineError("Each accepted teacher needs five calibration sources.")
    indices = [int(source["run_index"]) for source in sources]
    if indices != list(CALIBRATION_INDICES):
        raise StrictTeacherPipelineError("Calibration source indices must be 00--04.")
    identities = [str(source["run_id"]) for source in sources]
    if len(set(identities)) != 5:
        raise StrictTeacherPipelineError("Calibration run identities must be distinct.")
    return [dict(source) for source in sources]


def build_frozen_registry(
    selected: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    candidate_seed_artifact_path: Path | str,
    candidate_seed_list_sha256: str,
    strict_config_identity: Mapping[str, Any],
    candidate_audit_by_depth: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    depths: dict[str, Any] = {}
    for depth in DEPTHS:
        rows = list(selected.get(depth, []))
        if len(rows) != 5:
            raise StrictTeacherPipelineError(f"Depth {depth} does not have five teachers.")
        assignments = []
        for task_index, raw in enumerate(rows):
            if not bool(raw.get("accepted")):
                raise StrictTeacherPipelineError("Frozen assignment did not pass both screens.")
            sources = _calibration_sources(raw)
            seed = int(raw.get("assigned_task_seed", raw.get("teacher_seed")))
            assignment = {
                **dict(raw),
                "depth": depth,
                "task_index": task_index,
                "acceptance_order": task_index,
                "assigned_task_seed": seed,
                "teacher_seed": seed,
                "split_seed": int(raw["split_seed"]),
                "status": "accepted",
                "relevant_config_hash": canonical_sha256(strict_config_identity),
                "calibration_sources": sources,
                "calibration_source_identity": canonical_sha256(sources),
            }
            assignments.append(assignment)
        depth_payload = {
            "depth": depth,
            "required_tasks": 5,
            "assignments": assignments,
        }
        if candidate_audit_by_depth is not None:
            audit = []
            selected_by_candidate = {
                int(row["candidate_index"]): int(row["task_index"])
                for row in assignments
            }
            for raw in candidate_audit_by_depth.get(depth, []):
                candidate = dict(raw)
                candidate.update(candidate_decision(candidate, depth=depth))
                candidate["selected_task_index"] = selected_by_candidate.get(
                    int(candidate["candidate_index"])
                )
                audit.append(candidate)
            depth_payload["candidates"] = audit
        depths[str(depth)] = depth_payload
    payload = {
        "registry_schema": STRICT_REGISTRY_SCHEMA,
        "teacher_screening_policy": STRICT_POLICY,
        "selection_rule": "first_five_passing_in_declared_candidate_order",
        "candidate_seed_artifact_path": str(candidate_seed_artifact_path),
        "candidate_seed_list_sha256": str(candidate_seed_list_sha256),
        "strict_config_identity": dict(strict_config_identity),
        "strict_config_identity_sha256": canonical_sha256(strict_config_identity),
        "depths": depths,
    }
    validate_frozen_registry(payload)
    return payload


def validate_frozen_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("registry_schema") != STRICT_REGISTRY_SCHEMA:
        raise StrictTeacherPipelineError("Strict registry schema mismatch.")
    if payload.get("teacher_screening_policy") != STRICT_POLICY:
        raise StrictTeacherPipelineError("Strict registry policy mismatch.")
    if payload.get("selection_rule") != "first_five_passing_in_declared_candidate_order":
        raise StrictTeacherPipelineError("Strict registry selection rule mismatch.")
    seen_depth_seeds: set[tuple[int, int]] = set()
    for depth in DEPTHS:
        entry = payload.get("depths", {}).get(str(depth), {})
        rows = entry.get("assignments", [])
        if len(rows) != 5:
            raise StrictTeacherPipelineError(f"Strict registry depth {depth} needs five tasks.")
        if [int(row["task_index"]) for row in rows] != list(TASK_INDICES):
            raise StrictTeacherPipelineError(f"Strict registry depth {depth} task order is invalid.")
        candidate_indices = [int(row["candidate_index"]) for row in rows]
        if candidate_indices != sorted(candidate_indices):
            raise StrictTeacherPipelineError(
                f"Strict registry depth {depth} violates first-passing order."
            )
        for row in rows:
            if row.get("status") != "accepted" or not bool(row.get("signal_pass")):
                raise StrictTeacherPipelineError("Strict registry contains an unaccepted task.")
            if not bool(row.get("reference_learnability_pass")):
                raise StrictTeacherPipelineError("Strict registry contains an unlearnable task.")
            if float(row["median_reference_validation_accuracy"]) < LEARNABILITY_FLOORS[depth]:
                raise StrictTeacherPipelineError("Strict registry floor violation.")
            seed = int(row["assigned_task_seed"])
            depth_seed = (depth, seed)
            if depth_seed in seen_depth_seeds:
                raise StrictTeacherPipelineError(
                    "Strict registry contains a duplicate teacher seed within a depth."
                )
            seen_depth_seeds.add(depth_seed)
        candidates = entry.get("candidates")
        if candidates is not None:
            audit_indices = [int(row["candidate_index"]) for row in candidates]
            if audit_indices != sorted(audit_indices) or len(audit_indices) != len(
                set(audit_indices)
            ):
                raise StrictTeacherPipelineError(
                    f"Strict registry depth {depth} candidate audit order is invalid."
                )
            selected_indices = [
                int(row["candidate_index"])
                for row in candidates
                if row.get("selected_task_index") is not None
            ]
            if selected_indices != candidate_indices:
                raise StrictTeacherPipelineError(
                    f"Strict registry depth {depth} candidate audit disagrees "
                    "with first-five assignments."
                )
    return dict(payload)


def build_target_artifact(
    registry: Mapping[str, Any],
    *,
    registry_path: Path | str,
    registry_sha256: str,
) -> dict[str, Any]:
    validate_frozen_registry(registry)
    targets = []
    for depth in DEPTHS:
        for row in registry["depths"][str(depth)]["assignments"]:
            sources = _calibration_sources(row)
            accuracies = [float(source["validation_accuracy"]) for source in sources]
            median = calibration_median(accuracies)
            if median != float(row["median_reference_validation_accuracy"]):
                raise StrictTeacherPipelineError("Registry median differs from calibration sources.")
            targets.append(
                {
                    "depth": depth,
                    "task_index": int(row["task_index"]),
                    "target_key": f"depth_{depth}/task_{int(row['task_index']):02d}",
                    "original_candidate_index": int(row["candidate_index"]),
                    "teacher_seed": int(row["teacher_seed"]),
                    "assigned_task_seed": int(row["assigned_task_seed"]),
                    "split_seed": int(row["split_seed"]),
                    "calibration_run_indices": list(CALIBRATION_INDICES),
                    "calibration_sources": sources,
                    "calibration_validation_accuracies": accuracies,
                    "target_validation_accuracy": median,
                    "reference_learnability_floor": LEARNABILITY_FLOORS[depth],
                    "signal_metrics": dict(row["signal_metrics"]),
                    "screening_policy": STRICT_POLICY,
                    "calibration_source_identity": str(
                        row["calibration_source_identity"]
                    ),
                }
            )
    payload = {
        "schema_version": STRICT_TARGET_SCHEMA,
        "target_granularity": "depth_task",
        "target_strategy": "median_of_five_reference_student_validation_accuracies",
        "heldout_used_to_define_targets": False,
        "calibration_run_indices": list(CALIBRATION_INDICES),
        "evaluation_run_indices": list(HELDOUT_INDICES),
        "training_stopping_identity": dict(TRAINING_STOPPING_IDENTITY),
        "teacher_screening_policy": STRICT_POLICY,
        "teacher_task_registry_path": str(registry_path),
        "teacher_task_registry_sha256": str(registry_sha256),
        "candidate_seed_list_sha256": str(registry["candidate_seed_list_sha256"]),
        "n_depths": 5,
        "tasks_per_depth": 5,
        "n_targets": 25,
        "targets": targets,
    }
    validate_target_artifact(payload)
    return payload


def validate_target_artifact(
    payload: Mapping[str, Any], *, expected_registry_sha256: str | None = None
) -> dict[str, Any]:
    expected = {
        "schema_version": STRICT_TARGET_SCHEMA,
        "target_granularity": "depth_task",
        "heldout_used_to_define_targets": False,
        "calibration_run_indices": list(CALIBRATION_INDICES),
        "evaluation_run_indices": list(HELDOUT_INDICES),
        "n_depths": 5,
        "tasks_per_depth": 5,
        "n_targets": 25,
        "teacher_screening_policy": STRICT_POLICY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StrictTeacherPipelineError(f"Strict target {key} is incompatible.")
    registry_sha = str(payload.get("teacher_task_registry_sha256", ""))
    if len(registry_sha) != 64:
        raise StrictTeacherPipelineError("Strict target has no registry SHA-256.")
    if expected_registry_sha256 is not None and registry_sha != expected_registry_sha256:
        raise StrictTeacherPipelineError("Strict target registry SHA-256 mismatch.")
    rows = payload.get("targets")
    if not isinstance(rows, list) or len(rows) != 25:
        raise StrictTeacherPipelineError("Strict target artifact must contain 25 records.")
    observed = set()
    for row in rows:
        key = (int(row["depth"]), int(row["task_index"]))
        if key in observed:
            raise StrictTeacherPipelineError("Strict target contains a duplicate task.")
        observed.add(key)
        values = [float(value) for value in row["calibration_validation_accuracies"]]
        if len(values) != 5 or calibration_median(values) != float(
            row["target_validation_accuracy"]
        ):
            raise StrictTeacherPipelineError("Strict target is not a five-run median.")
        if float(row["target_validation_accuracy"]) < LEARNABILITY_FLOORS[key[0]]:
            raise StrictTeacherPipelineError("Strict target violates its admission floor.")
        sources = row["calibration_sources"]
        if canonical_sha256(sources) != row["calibration_source_identity"]:
            raise StrictTeacherPipelineError("Strict target calibration identity mismatch.")
    expected_keys = {(depth, task) for depth in DEPTHS for task in TASK_INDICES}
    if observed != expected_keys:
        raise StrictTeacherPipelineError("Strict target task coverage is incomplete.")
    return dict(payload)


def target_entry(
    payload: Mapping[str, Any], *, depth: int, task_index: int
) -> dict[str, Any]:
    validate_target_artifact(payload)
    for row in payload["targets"]:
        if int(row["depth"]) == int(depth) and int(row["task_index"]) == int(task_index):
            return dict(row)
    raise KeyError(f"No strict target for depth={depth}, task={task_index}.")


def build_accepted_stage_rows(
    *,
    registry: Mapping[str, Any],
    noise_levels: Sequence[float],
    boosting_seed_base: int = 410_000,
) -> dict[str, list[dict[str, Any]]]:
    """Build schema/count dry-run rows after the registry and targets are frozen."""

    validate_frozen_registry(registry)
    calibration: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    boosting: list[dict[str, Any]] = []
    for depth in DEPTHS:
        for task in TASK_INDICES:
            assignment = registry["depths"][str(depth)]["assignments"][task]
            common = {
                "depth": depth,
                "task_index": task,
                "candidate_index": int(assignment["candidate_index"]),
                "assigned_task_seed": int(assignment["assigned_task_seed"]),
                "split_seed": int(assignment["split_seed"]),
            }
            for run in CALIBRATION_INDICES:
                calibration.append({**common, "task_stage": "accepted_calibration_em", "em_run_index": run})
            for run in HELDOUT_INDICES:
                heldout.append({**common, "task_stage": "heldout_em", "em_run_index": run})
            for noise in noise_levels:
                for run in BOOSTING_INDICES:
                    boosting.append(
                        {
                            **common,
                            "task_stage": "adaboost_r",
                            "noise_strength": float(noise),
                            "boosting_run_index": run,
                            "boosting_seed": int(boosting_seed_base)
                            + depth * 1000
                            + task * 10
                            + run,
                        }
                    )
    return {"accepted_calibration": calibration, "heldout": heldout, "boosting": boosting}


def summarize_teacher_clusters(
    outcomes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Two-level variance decomposition: seeds within teacher, then teachers."""

    grouped: dict[tuple[int, float, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    success: dict[tuple[int, float, str], dict[int, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in outcomes:
        outer = (int(row["depth"]), float(row["noise_strength"]), str(row["method"]))
        task = int(row["task_index"])
        grouped[outer][task].append(float(row["cost"]))
        success[outer][task].append(bool(row["target_reached"]))
    summaries = []
    for key, task_values in sorted(grouped.items()):
        if set(task_values) != set(TASK_INDICES):
            raise StrictTeacherPipelineError(
                f"Hierarchy block {key} must contain five teacher tasks."
            )
        task_means = []
        within_variances = []
        task_success = []
        for task in TASK_INDICES:
            values = task_values[task]
            if len(values) != 5:
                raise StrictTeacherPipelineError(
                    f"Teacher {key + (task,)} must contain five optimizer paths."
                )
            task_means.append(float(np.mean(values)))
            within_variances.append(float(np.var(values, ddof=1)))
            task_success.append(float(np.mean(success[key][task])))
        summaries.append(
            {
                "depth": key[0],
                "noise_strength": key[1],
                "method": key[2],
                "n_teacher_tasks": 5,
                "optimizer_paths_per_teacher": 5,
                "teacher_task_means": task_means,
                "mean_cost_across_teacher_tasks": float(np.mean(task_means)),
                "between_teacher_variance": float(np.var(task_means, ddof=1)),
                "mean_within_teacher_optimizer_seed_variance": float(
                    np.mean(within_variances)
                ),
                "teacher_level_success_probabilities": task_success,
                "success_probability": float(np.mean(task_success)),
                "aggregation_unit": "teacher_task_cluster",
            }
        )
    return summaries


def minimal_unique_is_compatible(identity: Mapping[str, Any]) -> bool:
    """Strict reuse gate: legacy/minimal-unique identities are never compatible."""

    return (
        identity.get("teacher_screening_policy") == STRICT_POLICY
        and identity.get("teachers_per_depth") == 5
        and identity.get("accounting_schema_identity") == ACCOUNTING_SCHEMA
    )


__all__ = [
    "ACCOUNTING_SCHEMA",
    "BOOSTING_INDICES",
    "CALIBRATION_INDICES",
    "CANDIDATE_EXTENSION_POLICY",
    "CANDIDATE_SEED_SCHEMA",
    "CANDIDATE_SEED_SCHEMA_V2",
    "DEPTHS",
    "HELDOUT_INDICES",
    "LEARNABILITY_FLOORS",
    "SIGNAL_THRESHOLDS",
    "STRICT_POLICY",
    "STRICT_REGISTRY_SCHEMA",
    "STRICT_TARGET_SCHEMA",
    "StrictTeacherPipelineError",
    "build_accepted_stage_rows",
    "build_frozen_registry",
    "build_target_artifact",
    "calibration_median",
    "candidate_decision",
    "candidate_seed_parent_identity",
    "canonical_sha256",
    "file_sha256",
    "load_candidate_seed_artifact",
    "minimal_unique_is_compatible",
    "select_first_passing_candidates",
    "strict_identity_from_config",
    "strict_signal_summary",
    "summarize_teacher_clusters",
    "target_entry",
    "validate_frozen_registry",
    "validate_target_artifact",
]
