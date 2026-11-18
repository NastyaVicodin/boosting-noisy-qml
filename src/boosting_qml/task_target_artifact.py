"""Strict construction and validation of frozen per-(depth, task) targets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pec_boosting.depth_task_targets.patience20.v2"
TARGET_GRANULARITY = "depth_task"
TRAINING_STOPPING_IDENTITY = {
    "training_stopping_definition": "best_loss_no_improvement_patience.v1",
    "max_epochs": 500,
    "fixed_min_epochs": None,
    "early_stopping_patience_epochs": 20,
    "early_stopping_check_frequency_epochs": 1,
    "early_stopping_min_delta": 0.0,
    "monitored_loss": "ordinary_epoch_average_smooth_hinge_PsiG_sigma_0p5",
    "restore_best_parameters": True,
}
CALIBRATION_GLOB = "run_mu0p05_0[0-4].json"
EVALUATION_GLOB = "run_mu0p05_0[5-7].json"
CALIBRATION_INDICES = tuple(range(5))
EXPECTED_DEPTHS = tuple(range(1, 6))
EXPECTED_TASK_INDICES = tuple(range(10))
_RUN_INDEX_RE = re.compile(r"run_mu0p05_(?P<index>\d+)\.json$")


class TaskTargetArtifactError(ValueError):
    """Raised when target construction or validation is scientifically unsafe."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskTargetArtifactError(f"Unreadable calibration JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TaskTargetArtifactError(f"Calibration JSON is not an object: {path}")
    return payload


def _require_equal(observed: Any, expected: Any, *, field: str, context: str) -> None:
    if observed is None:
        raise TaskTargetArtifactError(f"{context} is missing {field}.")
    try:
        equal = int(observed) == int(expected)
    except (TypeError, ValueError):
        equal = observed == expected
    if not equal:
        raise TaskTargetArtifactError(
            f"{context} has {field}={observed!r}; expected {expected!r}."
        )


def _run_index(path: Path) -> int:
    match = _RUN_INDEX_RE.fullmatch(path.name)
    if match is None:
        raise TaskTargetArtifactError(f"Invalid calibration filename: {path}")
    return int(match.group("index"))


def calibration_source_identity(sources: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(sources), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def calibration_entry(
    *,
    depth: int,
    task_index: int,
    assignment: Mapping[str, Any],
    artifact_dir: Path,
    registry_path: Path,
    registry_sha256: str,
    calibration_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build one target entry after validating all five source runs."""

    dataset_path = artifact_dir / "configs" / "dataset_conf.json"
    dataset = _read_object(dataset_path)
    expected_seed = int(assignment["assigned_task_seed"])
    expected_split = int(assignment["split_seed"])
    expected_candidate = int(assignment["candidate_index"])
    context = f"depth={depth}, task={task_index}"
    for field, expected in (
        ("n_layers", depth),
        ("teacher_n_layers", depth),
        ("task_index", task_index),
        ("candidate_index", expected_candidate),
        ("data_generation_np_random_seed", expected_seed),
        ("validation_seed", expected_split),
    ):
        _require_equal(dataset.get(field), expected, field=field, context=str(dataset_path))
    if dataset.get("task_registry_sha256") != registry_sha256:
        raise TaskTargetArtifactError(f"{dataset_path} has a mismatched task registry SHA-256.")
    if Path(str(dataset.get("task_registry_path", ""))).resolve() != registry_path.resolve():
        raise TaskTargetArtifactError(f"{dataset_path} points to a different task registry.")
    if dataset.get("relevant_config_hash") != assignment.get("relevant_config_hash"):
        raise TaskTargetArtifactError(f"{dataset_path} has a mismatched relevant config hash.")

    paths = (
        sorted((artifact_dir / "em_runs").glob(CALIBRATION_GLOB))
        if calibration_paths is None
        else [Path(path) for path in calibration_paths]
    )
    indices = [_run_index(path) for path in paths]
    if len(paths) != 5 or sorted(indices) != list(CALIBRATION_INDICES):
        raise TaskTargetArtifactError(
            f"{context} requires exactly calibration runs 00-04; observed indices={indices}."
        )
    if len(set(indices)) != len(indices):
        raise TaskTargetArtifactError(f"{context} contains duplicate calibration run indices.")

    sources: list[dict[str, Any]] = []
    run_identities: set[str] = set()
    accuracies: list[float] = []
    for path, index in sorted(zip(paths, indices), key=lambda item: item[1]):
        payload = _read_object(path)
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise TaskTargetArtifactError(f"{path} is missing a non-empty run_id.")
        if run_id in run_identities:
            raise TaskTargetArtifactError(f"{context} contains duplicate run identity {run_id!r}.")
        run_identities.add(run_id)

        reproducibility = payload.get("reproducibility")
        if not isinstance(reproducibility, Mapping):
            raise TaskTargetArtifactError(f"{path} is missing reproducibility metadata.")
        _require_equal(
            reproducibility.get("data_generation_np_random_seed"),
            expected_seed,
            field="data_generation_np_random_seed",
            context=str(path),
        )
        _require_equal(
            reproducibility.get("validation_seed"),
            expected_split,
            field="validation_seed",
            context=str(path),
        )
        _require_equal(payload.get("n_layers"), depth, field="n_layers", context=str(path))
        metadata = payload.get("scientific_metadata")
        if isinstance(metadata, Mapping):
            for field, expected in (
                ("teacher_depth", depth),
                ("student_depth", depth),
                ("task_index", task_index),
                ("task_seed", expected_seed),
                ("split_seed", expected_split),
                ("task_registry_sha256", registry_sha256),
            ):
                if field in metadata:
                    _require_equal(
                        metadata.get(field), expected, field=field, context=str(path)
                    )

        em_payload = payload.get("em_payload_for_boosting")
        if not isinstance(em_payload, Mapping):
            raise TaskTargetArtifactError(f"{path} is missing em_payload_for_boosting.")
        accuracy = em_payload.get("reference_validation_accuracy")
        if accuracy is None:
            raise TaskTargetArtifactError(f"{path} is missing reference_validation_accuracy.")
        accuracy = float(accuracy)
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise TaskTargetArtifactError(
                f"{path} has invalid validation accuracy {accuracy!r}."
            )
        accuracies.append(accuracy)
        stopping = payload.get("training_stopping")
        if not isinstance(stopping, Mapping):
            raise TaskTargetArtifactError(f"{path} is missing training_stopping.")
        for field, expected in TRAINING_STOPPING_IDENTITY.items():
            observed = stopping.get(field)
            if expected is None:
                matches = observed in {None, 0}
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    matches = float(observed) == float(expected)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = observed == expected
            if not matches:
                raise TaskTargetArtifactError(
                    f"{path} has incompatible training stopping field "
                    f"{field}={observed!r}; expected {expected!r}."
                )
        sources.append(
            {
                "run_index": index,
                "run_id": run_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "validation_accuracy": accuracy,
                "data_generation_np_random_seed": int(
                    reproducibility["data_generation_np_random_seed"]
                ),
                "validation_seed": int(reproducibility["validation_seed"]),
                "training_stopping_definition": str(
                    stopping["training_stopping_definition"]
                ),
            }
        )

    target = float(statistics.median(accuracies))
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise TaskTargetArtifactError(f"{context} produced invalid median target {target!r}.")
    calibration_identity = calibration_source_identity(sources)
    return {
        "depth": int(depth),
        "task_index": int(task_index),
        "target_key": f"depth_{int(depth)}/task_{int(task_index):02d}",
        "assigned_task_seed": expected_seed,
        "split_seed": expected_split,
        "candidate_index": expected_candidate,
        "relevant_config_hash": str(assignment["relevant_config_hash"]),
        "dataset_config_path": str(dataset_path),
        "dataset_config_sha256": sha256_file(dataset_path),
        "calibration_run_indices": list(CALIBRATION_INDICES),
        "calibration_sources": sources,
        "calibration_validation_accuracies": accuracies,
        "target_validation_accuracy": target,
        "calibration_source_identity": calibration_identity,
    }


def build_target_artifact(
    *,
    config: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
    registry_path: Path,
    registry_sha256: str,
    artifact_for_task: Any,
    source_run_label: str,
    source_config_path: Path,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for depth in EXPECTED_DEPTHS:
        assignments = registry_payload.get("depths", {}).get(str(depth), {}).get(
            "assignments", []
        )
        by_task = {int(item["task_index"]): item for item in assignments}
        for task_index in EXPECTED_TASK_INDICES:
            assignment = by_task.get(task_index)
            if assignment is None or assignment.get("status") != "accepted":
                raise TaskTargetArtifactError(
                    f"Registry is missing accepted depth={depth}, task={task_index}."
                )
            entries.append(
                calibration_entry(
                    depth=depth,
                    task_index=task_index,
                    assignment=assignment,
                    artifact_dir=Path(artifact_for_task(depth, task_index)),
                    registry_path=registry_path,
                    registry_sha256=registry_sha256,
                )
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_granularity": TARGET_GRANULARITY,
        "target_strategy": "median",
        "target_percentile": 50.0,
        "calibration_glob": CALIBRATION_GLOB,
        "evaluation_glob": EVALUATION_GLOB,
        "calibration_run_indices": list(CALIBRATION_INDICES),
        "evaluation_run_indices": [5, 6, 7],
        "heldout_used_to_define_targets": False,
        "training_stopping_identity": dict(TRAINING_STOPPING_IDENTITY),
        "teacher_task_registry_path": str(registry_path),
        "teacher_task_registry_sha256": registry_sha256,
        "source_run_label": str(source_run_label),
        "source_config_path": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "source_provenance": dict(source_provenance),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_depths": len(EXPECTED_DEPTHS),
        "tasks_per_depth": len(EXPECTED_TASK_INDICES),
        "n_targets": len(entries),
        "targets": entries,
    }
    validate_target_artifact(payload)
    return payload


def validate_target_artifact(
    payload: Any,
    *,
    expected_registry_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TaskTargetArtifactError("Target artifact must be a JSON object.")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "target_granularity": TARGET_GRANULARITY,
        "target_strategy": "median",
        "target_percentile": 50.0,
        "calibration_glob": CALIBRATION_GLOB,
        "evaluation_glob": EVALUATION_GLOB,
        "heldout_used_to_define_targets": False,
        "n_depths": 5,
        "tasks_per_depth": 10,
        "n_targets": 50,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise TaskTargetArtifactError(
                f"Target artifact {key}={payload.get(key)!r}; expected {expected!r}."
            )
    if payload.get("calibration_run_indices") != list(CALIBRATION_INDICES):
        raise TaskTargetArtifactError("Target artifact calibration indices are not 00-04.")
    if payload.get("evaluation_run_indices") != [5, 6, 7]:
        raise TaskTargetArtifactError("Target artifact evaluation indices are not 05-07.")
    if payload.get("training_stopping_identity") != TRAINING_STOPPING_IDENTITY:
        raise TaskTargetArtifactError(
            "Target artifact has an incompatible training-stopping identity."
        )
    registry_sha = payload.get("teacher_task_registry_sha256")
    if not isinstance(registry_sha, str) or len(registry_sha) != 64:
        raise TaskTargetArtifactError("Target artifact has no valid registry SHA-256.")
    if expected_registry_sha256 is not None and registry_sha != expected_registry_sha256:
        raise TaskTargetArtifactError("Target artifact registry SHA-256 does not match.")

    entries = payload.get("targets")
    if not isinstance(entries, list) or len(entries) != 50:
        raise TaskTargetArtifactError("Target artifact must contain exactly 50 target entries.")
    observed_keys: set[tuple[int, int]] = set()
    calibration_identities: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TaskTargetArtifactError("Each target entry must be a JSON object.")
        depth = int(entry.get("depth", -1))
        task_index = int(entry.get("task_index", -1))
        key = (depth, task_index)
        if key in observed_keys:
            raise TaskTargetArtifactError(f"Duplicate target entry {key}.")
        observed_keys.add(key)
        if entry.get("target_key") != f"depth_{depth}/task_{task_index:02d}":
            raise TaskTargetArtifactError(f"Invalid target key for {key}.")
        if entry.get("calibration_run_indices") != list(CALIBRATION_INDICES):
            raise TaskTargetArtifactError(f"Target {key} does not use exactly runs 00-04.")
        sources = entry.get("calibration_sources")
        values = entry.get("calibration_validation_accuracies")
        if not isinstance(sources, list) or len(sources) != 5:
            raise TaskTargetArtifactError(f"Target {key} must have five calibration sources.")
        if not isinstance(values, list) or len(values) != 5:
            raise TaskTargetArtifactError(f"Target {key} must have five calibration values.")
        if [source.get("run_index") for source in sources] != list(CALIBRATION_INDICES):
            raise TaskTargetArtifactError(f"Target {key} source indices are not 00-04.")
        run_ids = [source.get("run_id") for source in sources]
        if len(set(run_ids)) != 5:
            raise TaskTargetArtifactError(f"Target {key} has duplicate calibration identities.")
        numeric_values = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric_values):
            raise TaskTargetArtifactError(f"Target {key} has invalid calibration values.")
        assigned_seed = int(entry.get("assigned_task_seed"))
        split_seed = int(entry.get("split_seed"))
        for source_index, (source, expected_value) in enumerate(
            zip(sources, numeric_values)
        ):
            if not isinstance(source, Mapping):
                raise TaskTargetArtifactError(f"Target {key} has a non-object source.")
            source_accuracy = float(source.get("validation_accuracy"))
            if source_accuracy != expected_value:
                raise TaskTargetArtifactError(
                    f"Target {key} source/value mismatch at run {source_index}."
                )
            if int(source.get("data_generation_np_random_seed")) != assigned_seed:
                raise TaskTargetArtifactError(
                    f"Target {key} source task seed mismatch at run {source_index}."
                )
            if int(source.get("validation_seed")) != split_seed:
                raise TaskTargetArtifactError(
                    f"Target {key} source split seed mismatch at run {source_index}."
                )
            if (
                source.get("training_stopping_definition")
                != TRAINING_STOPPING_IDENTITY["training_stopping_definition"]
            ):
                raise TaskTargetArtifactError(
                    f"Target {key} source stopping definition mismatch at run "
                    f"{source_index}."
                )
            if not isinstance(source.get("path"), str) or not source["path"]:
                raise TaskTargetArtifactError(f"Target {key} has an invalid source path.")
            if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
                raise TaskTargetArtifactError(f"Target {key} has an invalid source SHA-256.")
        target = float(entry.get("target_validation_accuracy"))
        if (
            not math.isfinite(target)
            or not 0.0 <= target <= 1.0
            or target != float(statistics.median(numeric_values))
        ):
            raise TaskTargetArtifactError(f"Target {key} is not the calibration median.")
        identity = entry.get("calibration_source_identity")
        if not isinstance(identity, str) or len(identity) != 64:
            raise TaskTargetArtifactError(f"Target {key} has invalid calibration identity.")
        if identity != calibration_source_identity(sources):
            raise TaskTargetArtifactError(
                f"Target {key} calibration source identity does not match its sources."
            )
        if identity in calibration_identities:
            raise TaskTargetArtifactError("Duplicate calibration source identity across tasks.")
        calibration_identities.add(identity)
    expected_keys = {
        (depth, task_index)
        for depth in EXPECTED_DEPTHS
        for task_index in EXPECTED_TASK_INDICES
    }
    if observed_keys != expected_keys:
        raise TaskTargetArtifactError(
            f"Target artifact task coverage differs: missing={sorted(expected_keys-observed_keys)} "
            f"extra={sorted(observed_keys-expected_keys)}."
        )
    return dict(payload)


def target_entry(payload: Mapping[str, Any], *, depth: int, task_index: int) -> dict[str, Any]:
    validate_target_artifact(payload)
    for entry in payload["targets"]:
        if int(entry["depth"]) == int(depth) and int(entry["task_index"]) == int(task_index):
            return dict(entry)
    raise KeyError(f"No target for depth={depth}, task_index={task_index}.")


def atomic_write_target_artifact(path: Path | str, payload: Any) -> Path:
    validated = validate_target_artifact(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(validated, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        reparsed = json.loads(destination.read_text(encoding="utf-8"))
        validate_target_artifact(reparsed)
        if reparsed != json.loads(rendered):
            raise TaskTargetArtifactError("Reparsed target artifact differs from input.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
