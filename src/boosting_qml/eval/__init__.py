"""Evaluation and artifact-level utilities."""

try:
    from .metrics import Batch, Result, get_accuracy, get_metrics
except Exception:  # pragma: no cover - JAX not installed in lightweight test env
    Batch = None  # type: ignore[assignment]
    Result = None  # type: ignore[assignment]
    get_accuracy = None  # type: ignore[assignment]
    get_metrics = None  # type: ignore[assignment]

from .em_artifacts import (
    aggregate_em_run_targets,
    build_boosting_stop_target,
    build_boosting_stop_target_from_run_summaries,
    build_boosting_stop_target_from_summary,
    build_boosting_target_candidates,
    list_em_artifact_paths,
    load_and_aggregate_em_runs,
    load_em_run_summaries,
    load_em_run_summaries_from_dir,
    load_em_run_summary,
    resolve_em_target_from_aggregation,
)

__all__ = [
    'Batch',
    'Result',
    'aggregate_em_run_targets',
    'build_boosting_stop_target',
    'build_boosting_stop_target_from_run_summaries',
    'build_boosting_stop_target_from_summary',
    'build_boosting_target_candidates',
    'get_accuracy',
    'get_metrics',
    'list_em_artifact_paths',
    'load_and_aggregate_em_runs',
    'load_em_run_summaries',
    'load_em_run_summaries_from_dir',
    'load_em_run_summary',
    'resolve_em_target_from_aggregation',
]
