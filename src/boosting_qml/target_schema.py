"""Frozen target schema helpers with depth- and task-scoped compatibility."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping


def load_target_payload(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Target source must be a JSON object: {path}")
    return payload


def target_value(
    payload: Mapping[str, Any],
    *,
    depth: int,
    task_index: int | None,
) -> float:
    """Resolve either the legacy depth target or the proposed depth/task target."""

    targets = payload.get("targets")
    epsilon_targets = payload.get("epsilon_mar_by_depth_task")
    if isinstance(epsilon_targets, Mapping):
        if task_index is None:
            raise ValueError("task_index is required for a task-specific margin target source.")
        key = f"depth_{int(depth)}_task_{int(task_index):02d}"
        if key not in epsilon_targets:
            raise KeyError(f"No frozen margin target for {key}.")
        return float(epsilon_targets[key])
    if isinstance(targets, list):
        if payload.get("schema_version") == (
            "pec_boosting.depth_task_targets.strict_teachers_5x5.v1"
        ):
            from boosting_qml.strict_teacher_pipeline import target_entry
        else:
            from boosting_qml.task_target_artifact import target_entry

        if task_index is None:
            raise ValueError("task_index is required for a task-specific target source.")
        return float(
            target_entry(payload, depth=int(depth), task_index=int(task_index))[
                "target_validation_accuracy"
            ]
        )
    if "targets_by_depth_task" in payload:
        if task_index is None:
            raise ValueError("task_index is required for a task-specific target source.")
        by_depth = payload["targets_by_depth_task"]
        if not isinstance(by_depth, Mapping):
            raise ValueError("targets_by_depth_task must be a JSON object.")
        task_map = by_depth.get(str(int(depth)))
        if not isinstance(task_map, Mapping):
            raise KeyError(f"No task-specific targets for depth {depth}.")
        task_key = f"task_{int(task_index):02d}"
        if task_key not in task_map:
            raise KeyError(f"No task-specific target for depth {depth}, {task_key}.")
        return float(task_map[task_key])

    if not isinstance(targets, Mapping):
        raise ValueError("Target source has neither targets nor targets_by_depth_task.")
    return float(targets[str(int(depth))])


def target_value_from_file(
    path: Path | str,
    *,
    depth: int,
    task_index: int | None,
) -> float:
    return target_value(load_target_payload(path), depth=depth, task_index=task_index)


def target_identity_from_file(
    path: Path | str,
    *,
    depth: int,
    task_index: int,
    assigned_task_seed: int,
) -> dict[str, Any]:
    target_path = Path(path)
    payload = load_target_payload(target_path)
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    targets = payload.get("targets")
    epsilon_targets = payload.get("epsilon_mar_by_depth_task")
    if isinstance(epsilon_targets, Mapping):
        selected = target_value(payload, depth=depth, task_index=task_index)
        return {
            "target_granularity": "depth_task",
            "target_depth": int(depth),
            "target_task_index": int(task_index),
            "target_assigned_task_seed": int(assigned_task_seed),
            "selected_target_value": float(selected),
            "target_artifact_path": str(target_path),
            "target_artifact_sha256": digest,
            "target_schema_version": str(payload["schema_version"]),
            "calibration_source_identity": (
                f"calibration_margin:{payload.get('calibration_source_sha256', digest)}"
            ),
            "teacher_task_registry_sha256": None,
        }
    if isinstance(targets, list):
        if payload.get("schema_version") == (
            "pec_boosting.depth_task_targets.strict_teachers_5x5.v1"
        ):
            from boosting_qml.strict_teacher_pipeline import target_entry
        else:
            from boosting_qml.task_target_artifact import target_entry

        entry = target_entry(payload, depth=depth, task_index=task_index)
        if int(entry["assigned_task_seed"]) != int(assigned_task_seed):
            raise ValueError(
                f"Target artifact task seed differs for depth={depth}, task={task_index}."
            )
        return {
            "target_granularity": str(payload["target_granularity"]),
            "target_depth": int(depth),
            "target_task_index": int(task_index),
            "target_assigned_task_seed": int(entry["assigned_task_seed"]),
            "selected_target_value": float(entry["target_validation_accuracy"]),
            "target_artifact_path": str(target_path),
            "target_artifact_sha256": digest,
            "target_schema_version": str(payload["schema_version"]),
            "calibration_source_identity": str(entry["calibration_source_identity"]),
            "teacher_task_registry_sha256": str(
                payload["teacher_task_registry_sha256"]
            ),
        }
    return {
        "target_granularity": "depth",
        "target_depth": int(depth),
        "target_task_index": None,
        "target_assigned_task_seed": None,
        "selected_target_value": target_value(
            payload, depth=depth, task_index=task_index
        ),
        "target_artifact_path": str(target_path),
        "target_artifact_sha256": digest,
        "target_schema_version": str(payload.get("schema_version", "legacy.depth_targets.v1")),
        "calibration_source_identity": f"depth:{int(depth)}:{digest}",
    }
