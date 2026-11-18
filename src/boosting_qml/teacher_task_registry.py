"""Deterministic teacher-task screening and frozen registry support.

The registry is the enforcement mechanism for the ``minimal_unique`` policy.
It deliberately hashes only dataset-defining configuration so that projected,
physical-validation, and appendix stages can share identical teacher tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from boosting_qml.data.dataset_scaling import scaled_teacher_student_dataset_sizes


REGISTRY_SCHEMA = "teacher_task_registry.v1"
SCREENING_POLICY = "minimal_unique"
STRICT_SCREENING_POLICY = "strict_signal_then_reference_learnability.first5.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TaskRegistryError(ValueError):
    """Raised when a required teacher-task registry is absent or incompatible."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _grid(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("grid_documentation") or config.get("grid") or {})


def _training(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("training") or {})


def screening_policy(config: Mapping[str, Any]) -> str:
    return str(
        config.get(
            "teacher_screening_policy",
            dict(config.get("scientific_protocol") or {}).get("teacher_screening_policy", ""),
        )
    )


def uses_frozen_registry(config: Mapping[str, Any]) -> bool:
    return screening_policy(config) in {SCREENING_POLICY, STRICT_SCREENING_POLICY}


def registry_path(config: Mapping[str, Any], override: Path | str | None = None) -> Path:
    value = override if override is not None else config.get("task_registry_path")
    if value in {None, ""}:
        raise TaskRegistryError(
            "teacher_screening_policy='minimal_unique' requires task_registry_path "
            "or --registry."
        )
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def tasks_per_depth(config: Mapping[str, Any]) -> int:
    grid = _grid(config)
    if grid.get("task_indices") is not None:
        indices = [int(value) for value in grid["task_indices"]]
        return max(indices) + 1 if indices else 0
    return int(
        grid.get(
            "independent_tasks_per_depth",
            grid.get("tasks_per_depth", grid.get("independent_tasks", 1)),
        )
    )


def configured_depths(config: Mapping[str, Any]) -> list[int]:
    return [int(value) for value in _grid(config).get("depths", [])]


def _auto_scale_enabled(config: Mapping[str, Any]) -> bool:
    grid = _grid(config)
    return bool(grid.get("auto_scale_dataset", config.get("auto_scale_dataset", True)))


def resolve_dataset_spec(config: Mapping[str, Any], depth: int) -> dict[str, Any]:
    """Resolve the exact production dataset sizes and teacher-generation knobs."""
    grid = _grid(config)
    training = _training(config)
    n_qubits = int(grid.get("n_qubits", config.get("n_qubits", 4)))
    batch_size = int(training.get("batch_size", grid.get("batch_size", 64)))
    validation_fraction = float(grid.get("validation_fraction", config.get("validation_fraction", 0.2)))
    min_grid_points = int(
        grid.get(
            "teacher_grid_points",
            grid.get("min_teacher_grid_points", config.get("teacher_grid_points", 10_000)),
        )
    )

    if _auto_scale_enabled(config):
        scaling = scaled_teacher_student_dataset_sizes(
            n_qubits=n_qubits,
            student_depth=int(depth),
            validation_fraction=validation_fraction,
            batch_size=batch_size,
            min_fit_train_size=int(grid.get("min_fit_train_size", 1536)),
            samples_per_student_parameter=float(grid.get("samples_per_student_parameter", 10.0)),
            samples_per_qubit=int(grid.get("samples_per_qubit", 256)),
            min_test_size=int(grid.get("min_test_size", 1024)),
            test_size_ratio=float(grid.get("test_size_ratio", 0.5)),
            min_teacher_grid_points=min_grid_points,
            teacher_grid_multiplier=int(grid.get("teacher_grid_multiplier", 20)),
            scale_depth=grid.get("dataset_scale_depth", 5),
        )
        train_size = int(scaling["train_size"])
        test_size = int(scaling["test_size"])
        teacher_grid_size = int(scaling["teacher_n_grid_points"])
    else:
        train_size = int(grid.get("train_size", config.get("train_size", 1024)))
        test_size = int(grid.get("test_size", config.get("test_size", 512)))
        teacher_grid_size = min_grid_points
        scaling = {
            "mode": "manual",
            "train_size": train_size,
            "test_size": test_size,
            "teacher_n_grid_points": teacher_grid_size,
        }

    no_doubles = bool(config.get("teacher_student_no_doubles", True))
    required_per_class = int(math.ceil((train_size + test_size) / 2.0)) if no_doubles else 1
    return {
        "n_qubits": n_qubits,
        "teacher_depth": int(depth),
        "student_depth": int(depth),
        "teacher_margin": float(config.get("teacher_margin", 0.2)),
        "effective_train_size": train_size,
        "effective_test_size": test_size,
        "teacher_grid_size": teacher_grid_size,
        "required_unique_candidates_per_class": required_per_class,
        "no_doubles": no_doubles,
        "observable": "global_z",
        "probability_convention": "(global_z_expectation+1)/2",
        "teacher_circuit": "legacy_reuploading",
        "teacher_parameter_generation": "jax_prng_split_normal_float32",
        "latin_hypercube_seed_source": "task_seed",
        "dataset_scaling": scaling,
    }


def relevant_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return depth-independent dataset-defining configuration."""
    grid = _grid(config)
    training = _training(config)
    return {
        "screening_policy": SCREENING_POLICY,
        "n_qubits": int(grid.get("n_qubits", config.get("n_qubits", 4))),
        "teacher_margin": float(config.get("teacher_margin", 0.2)),
        "no_doubles": bool(config.get("teacher_student_no_doubles", True)),
        "auto_scale_dataset": _auto_scale_enabled(config),
        "dataset_scale_depth": int(grid.get("dataset_scale_depth", 5)),
        "validation_fraction": float(grid.get("validation_fraction", config.get("validation_fraction", 0.2))),
        "batch_size": int(training.get("batch_size", grid.get("batch_size", 64))),
        "min_fit_train_size": int(grid.get("min_fit_train_size", 1536)),
        "samples_per_student_parameter": float(grid.get("samples_per_student_parameter", 10.0)),
        "samples_per_qubit": int(grid.get("samples_per_qubit", 256)),
        "min_test_size": int(grid.get("min_test_size", 1024)),
        "test_size_ratio": float(grid.get("test_size_ratio", 0.5)),
        "min_teacher_grid_points": int(
            grid.get(
                "teacher_grid_points",
                grid.get("min_teacher_grid_points", config.get("teacher_grid_points", 10_000)),
            )
        ),
        "teacher_grid_multiplier": int(grid.get("teacher_grid_multiplier", 20)),
        "task_seed_base": int(config.get("task_seed_base", 100_000)),
        "split_seed_base": int(config.get("split_seed_base", 200_000)),
        "teacher_circuit": "legacy_reuploading",
        "observable": "global_z",
        "probability_convention": "(global_z_expectation+1)/2",
    }


def relevant_config_hash(config: Mapping[str, Any]) -> str:
    return _sha256(relevant_config_payload(config))


def depth_config_hash(config: Mapping[str, Any], depth: int) -> str:
    return _sha256(resolve_dataset_spec(config, depth))


def _source_metadata() -> dict[str, Any]:
    def run(command: list[str]) -> str:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    commit = run(["git", "rev-parse", "HEAD"])
    diff = run(["git", "diff", "--binary", "HEAD", "--", "src", "configs", "tests"])
    source_hasher = hashlib.sha256()
    for root_name in ("src", "configs"):
        root = PROJECT_ROOT / root_name
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_hasher.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
                source_hasher.update(b"\0")
                source_hasher.update(path.read_bytes())
                source_hasher.update(b"\0")
    return {
        "git_commit": commit or None,
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "source_tree_sha256": source_hasher.hexdigest(),
    }


CandidateEvaluator = Callable[[dict[str, Any], int], Mapping[str, int]]


def evaluate_teacher_candidate(spec: dict[str, Any], task_seed: int) -> dict[str, int]:
    """Evaluate class counts through the production teacher circuit conventions."""
    from boosting_qml.pqc.teacher_student import teacher_class_counts

    return teacher_class_counts(
        n_qubits=int(spec["n_qubits"]),
        n_layers=int(spec["teacher_depth"]),
        n_grid_points=int(spec["teacher_grid_size"]),
        margin=float(spec["teacher_margin"]),
        data_generation_seed=int(task_seed),
    )


def _candidate_record(
    *,
    config: Mapping[str, Any],
    depth: int,
    candidate_index: int,
    task_index: int | None,
    evaluator: CandidateEvaluator,
) -> dict[str, Any]:
    spec = resolve_dataset_spec(config, depth)
    task_seed = int(config.get("task_seed_base", 100_000)) + int(depth) * 100 + int(candidate_index)
    split_slot = candidate_index if task_index is None else task_index
    split_seed = int(config.get("split_seed_base", 200_000)) + int(depth) * 100 + int(split_slot)
    counts = evaluator(spec, task_seed)
    negative = int(counts["negative_candidate_count"])
    positive = int(counts["positive_candidate_count"])
    rejected = int(counts["rejected_margin_count"])
    required = int(spec["required_unique_candidates_per_class"])
    if negative == 0 or positive == 0:
        status = "rejected"
        reason = "one_class_empty_after_margin_thresholding"
    elif bool(spec["no_doubles"]) and min(negative, positive) < required:
        status = "rejected"
        reason = "insufficient_unique_points_per_class"
    else:
        status = "accepted"
        reason = None
    return {
        "depth": int(depth),
        "task_index": None if task_index is None else int(task_index),
        "assigned_task_seed": task_seed,
        "candidate_index": int(candidate_index),
        "split_seed": split_seed,
        "status": status,
        "rejection_reason": reason,
        "negative_candidate_count": negative,
        "positive_candidate_count": positive,
        "rejected_margin_count": rejected,
        "required_unique_candidates_per_class": required,
        "effective_train_size": int(spec["effective_train_size"]),
        "effective_test_size": int(spec["effective_test_size"]),
        "teacher_grid_size": int(spec["teacher_grid_size"]),
        "relevant_config_hash": depth_config_hash(config, depth),
    }


def build_registry(
    config: Mapping[str, Any],
    *,
    evaluator: CandidateEvaluator = evaluate_teacher_candidate,
) -> dict[str, Any]:
    """Build a deterministic slot-preserving minimal-unique registry."""
    if screening_policy(config) != SCREENING_POLICY:
        raise TaskRegistryError(
            f"prepare-task-registry requires teacher_screening_policy={SCREENING_POLICY!r}."
        )
    n_tasks = tasks_per_depth(config)
    if n_tasks <= 0:
        raise TaskRegistryError("At least one task per depth is required.")

    source = _source_metadata()
    depth_payloads: dict[str, Any] = {}
    for depth in configured_depths(config):
        candidates: list[dict[str, Any]] = []
        assignments: dict[int, dict[str, Any]] = {}
        invalid_slots: list[int] = []
        for candidate_index in range(n_tasks):
            record = _candidate_record(
                config=config,
                depth=depth,
                candidate_index=candidate_index,
                task_index=candidate_index,
                evaluator=evaluator,
            )
            candidates.append(record)
            if record["status"] == "accepted":
                assignments[candidate_index] = dict(record)
            else:
                invalid_slots.append(candidate_index)

        candidate_index = n_tasks
        for task_index in invalid_slots:
            while True:
                record = _candidate_record(
                    config=config,
                    depth=depth,
                    candidate_index=candidate_index,
                    task_index=None,
                    evaluator=evaluator,
                )
                candidate_index += 1
                if record["status"] == "accepted":
                    assigned = dict(record)
                    assigned["task_index"] = task_index
                    assigned["split_seed"] = (
                        int(config.get("split_seed_base", 200_000)) + int(depth) * 100 + task_index
                    )
                    candidates.append(assigned)
                    assignments[task_index] = assigned
                    break
                candidates.append(record)

        depth_payloads[str(depth)] = {
            "depth": int(depth),
            "required_tasks": n_tasks,
            "relevant_config_hash": depth_config_hash(config, depth),
            "source": source,
            "assignments": [assignments[index] for index in range(n_tasks)],
            "candidates": candidates,
        }

    return {
        "registry_schema": REGISTRY_SCHEMA,
        "teacher_screening_policy": SCREENING_POLICY,
        "config_hash": relevant_config_hash(config),
        "relevant_config": relevant_config_payload(config),
        "source": source,
        "depths": depth_payloads,
    }


def write_registry(path: Path, registry: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    return path


def registry_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(
    config: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    required_depths: list[int] | None = None,
    required_task_indices: list[int] | None = None,
) -> tuple[dict[str, Any], Path, str]:
    resolved = registry_path(config, path)
    if not resolved.is_file():
        raise TaskRegistryError(f"Required teacher-task registry does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("registry_schema") == (
        "teacher_task_registry.strict_signal_reference_first5.v1"
    ):
        from boosting_qml.strict_teacher_pipeline import (
            load_candidate_seed_artifact,
            strict_identity_from_config,
            validate_frozen_registry,
        )

        if screening_policy(config) != STRICT_SCREENING_POLICY:
            raise TaskRegistryError(
                "A strict teacher registry requires the strict screening policy."
            )
        try:
            validate_frozen_registry(payload)
        except ValueError as exc:
            raise TaskRegistryError(str(exc)) from exc
        candidate_path = config.get("candidate_seed_artifact_path")
        if not candidate_path:
            raise TaskRegistryError(
                "Strict teacher configuration is missing candidate_seed_artifact_path."
            )
        _, expected_candidate_sha = load_candidate_seed_artifact(candidate_path)
        if payload.get("candidate_seed_list_sha256") != expected_candidate_sha:
            raise TaskRegistryError(
                "Strict registry candidate-seed-list SHA-256 is incompatible."
            )
        assignment_reuse = config.get("strict_registry_assignment_reuse")
        if assignment_reuse is not None:
            if not isinstance(assignment_reuse, Mapping):
                raise TaskRegistryError("strict_registry_assignment_reuse must be an object.")
            if assignment_reuse.get("scope") != "teacher_task_and_split_only":
                raise TaskRegistryError(
                    "Strict registry assignment reuse must be scoped to "
                    "teacher_task_and_split_only."
                )
            expected_identity_sha = assignment_reuse.get(
                "source_strict_config_identity_sha256"
            )
            if expected_identity_sha != payload.get("strict_config_identity_sha256"):
                raise TaskRegistryError(
                    "Strict registry assignment-reuse identity SHA-256 is incompatible."
                )
            if assignment_reuse.get("attestation") != (
                "Reuse only the frozen accepted teacher identity and strict-v2 "
                "task/split assignment; do not reuse gradient, stopping, seed, or "
                "resource outcomes."
            ):
                raise TaskRegistryError("Strict registry assignment-reuse attestation is missing.")
        else:
            expected_strict_identity = strict_identity_from_config(
                config, candidate_seed_list_sha256=expected_candidate_sha
            )
            if payload.get("strict_config_identity") != expected_strict_identity:
                raise TaskRegistryError(
                    "Strict registry screening/floor/seed/stopping/noise/accounting "
                    "identity is incompatible with the configuration."
                )
        depths = (
            configured_depths(config)
            if required_depths is None
            else [int(value) for value in required_depths]
        )
        task_ids = (
            list(range(tasks_per_depth(config)))
            if required_task_indices is None
            else [int(value) for value in required_task_indices]
        )
        for depth in depths:
            rows = payload["depths"][str(depth)]["assignments"]
            by_task = {int(row["task_index"]): row for row in rows}
            missing = [task for task in task_ids if task not in by_task]
            if missing:
                raise TaskRegistryError(
                    f"Strict registry depth {depth} is missing task slots {missing}."
                )
        return payload, resolved, registry_sha256(resolved)
    if payload.get("registry_schema") != REGISTRY_SCHEMA:
        raise TaskRegistryError(
            f"Registry {resolved} has schema {payload.get('registry_schema')!r}; "
            f"expected {REGISTRY_SCHEMA!r}."
        )
    if payload.get("teacher_screening_policy") != SCREENING_POLICY:
        raise TaskRegistryError(f"Registry {resolved} does not enforce {SCREENING_POLICY!r}.")
    expected_hash = relevant_config_hash(config)
    if payload.get("config_hash") != expected_hash:
        raise TaskRegistryError(
            f"Registry config hash is incompatible: observed={payload.get('config_hash')} "
            f"expected={expected_hash}."
        )

    depths = configured_depths(config) if required_depths is None else [int(value) for value in required_depths]
    task_ids = (
        list(range(tasks_per_depth(config)))
        if required_task_indices is None
        else [int(value) for value in required_task_indices]
    )
    for depth in depths:
        depth_entry = dict(payload.get("depths", {}).get(str(depth)) or {})
        assignments = list(depth_entry.get("assignments") or [])
        accepted_assignments = [item for item in assignments if item.get("status") == "accepted"]
        required_count = int(depth_entry.get("required_tasks", len(assignments)))
        if len(accepted_assignments) < required_count:
            raise TaskRegistryError(
                f"Registry depth {depth} has fewer than its declared {required_count} accepted tasks."
            )
        all_task_slots = [int(item["task_index"]) for item in accepted_assignments]
        if len(all_task_slots) != len(set(all_task_slots)):
            raise TaskRegistryError(f"Registry depth {depth} contains duplicate task slots.")
        all_seeds = [int(item["assigned_task_seed"]) for item in accepted_assignments]
        if len(all_seeds) != len(set(all_seeds)):
            raise TaskRegistryError(f"Registry depth {depth} contains duplicate assigned task seeds.")

        expected_depth_hash = depth_config_hash(config, depth)
        if depth_entry.get("relevant_config_hash") != expected_depth_hash:
            raise TaskRegistryError(f"Registry depth {depth} has an incompatible dataset config hash.")
        by_task = {int(item["task_index"]): item for item in assignments}
        missing = [task for task in task_ids if task not in by_task or by_task[task].get("status") != "accepted"]
        if missing:
            raise TaskRegistryError(
                f"Registry depth {depth} has fewer than the configured accepted tasks; missing slots {missing}."
            )
        seeds = [int(by_task[task]["assigned_task_seed"]) for task in task_ids]
        if len(seeds) != len(set(seeds)):
            raise TaskRegistryError(f"Registry depth {depth} contains duplicate assigned task seeds.")
    return payload, resolved, registry_sha256(resolved)


def assignment(config: Mapping[str, Any], depth: int, task_index: int) -> dict[str, Any]:
    payload, _, _ = load_registry(
        config,
        required_depths=[int(depth)],
        required_task_indices=[int(task_index)],
    )
    assignments = payload["depths"][str(int(depth))]["assignments"]
    for item in assignments:
        if int(item["task_index"]) == int(task_index):
            return dict(item)
    raise TaskRegistryError(f"Registry has no assignment for depth={depth}, task_index={task_index}.")


def validate_manifest_rows(config: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> None:
    if not uses_frozen_registry(config):
        return
    _, path, digest = load_registry(config)
    for row in rows:
        depth = row.get("depth")
        task_index = row.get("task_index")
        if depth in {None, ""} or task_index in {None, ""}:
            if str(row.get("task_registry_sha256", "")) != digest:
                raise TaskRegistryError("Manifest row registry SHA256 differs from the validated registry.")
            continue
        item = assignment(config, int(depth), int(task_index))
        observed_seed = row.get("task_seed", row.get("assigned_task_seed"))
        if int(observed_seed) != int(item["assigned_task_seed"]):
            raise TaskRegistryError(
                f"Manifest task seed differs from registry for depth={depth}, task_index={task_index}: "
                f"observed={observed_seed}, expected={item['assigned_task_seed']}."
            )
        if int(row.get("split_seed")) != int(item["split_seed"]):
            raise TaskRegistryError(
                f"Manifest split seed differs from registry for depth={depth}, task_index={task_index}."
            )
        if str(row.get("task_registry_sha256", "")) != digest:
            raise TaskRegistryError("Manifest row registry SHA256 differs from the validated registry.")
        row_registry_path = Path(str(row.get("task_registry_path", "")))
        if not row_registry_path.is_absolute():
            row_registry_path = PROJECT_ROOT / row_registry_path
        if row_registry_path.resolve() != path.resolve():
            raise TaskRegistryError("Manifest row task_registry_path differs from configuration.")


__all__ = [
    "REGISTRY_SCHEMA",
    "SCREENING_POLICY",
    "STRICT_SCREENING_POLICY",
    "TaskRegistryError",
    "assignment",
    "build_registry",
    "configured_depths",
    "depth_config_hash",
    "evaluate_teacher_candidate",
    "load_registry",
    "registry_path",
    "registry_sha256",
    "relevant_config_hash",
    "resolve_dataset_spec",
    "screening_policy",
    "tasks_per_depth",
    "uses_frozen_registry",
    "validate_manifest_rows",
    "write_registry",
]
