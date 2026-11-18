"""Load and aggregate independent error-mitigation run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from boosting_qml.config import BoostingStopTargetConfig, EMRunArtifactConfig, EMTargetAggregationConfig
from boosting_qml.eval.em_artifact_compat import normalize_reference_diagnostics_v1
from boosting_qml.types import EMRunSummary, EMTargetAggregationSummary


_TARGET_FIELD_AND_ORIENTATION = {
    'margin_loss': ('target_margin_loss', True),
    'classification_error': ('target_classification_error', True),
    'mean_margin': ('target_mean_margin', False),
    'test_accuracy': ('reference_test_accuracy', False),
    'validation_accuracy': ('reference_validation_accuracy', False),
}


def _first_present(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _to_float(value: object, name: str, source_path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse float for '{name}' from {source_path}.") from exc


def _to_int(value: object, name: str, source_path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse int for '{name}' from {source_path}.") from exc


def _resolve_required_float(
    payload: Mapping[str, object],
    record: Mapping[str, object],
    *,
    payload_keys: Sequence[str],
    record_keys: Sequence[str],
    name: str,
    source_path: Path,
) -> float:
    value = _first_present(payload, payload_keys)
    if value is None:
        value = _first_present(record, record_keys)
    if value is None:
        raise KeyError(
            f"Missing required key for '{name}' in {source_path}. "
            f"Tried payload keys={list(payload_keys)} and record keys={list(record_keys)}."
        )
    return _to_float(value, name=name, source_path=source_path)


def _resolve_optional_float(
    payload: Mapping[str, object],
    record: Mapping[str, object],
    *,
    payload_keys: Sequence[str],
    record_keys: Sequence[str],
    default_value: float,
    name: str,
    source_path: Path,
) -> float:
    value = _first_present(payload, payload_keys)
    if value is None:
        value = _first_present(record, record_keys)
    if value is None:
        return float(default_value)
    return _to_float(value, name=name, source_path=source_path)


def _resolve_optional_float_or_none(
    payload: Mapping[str, object],
    record: Mapping[str, object],
    *,
    payload_keys: Sequence[str],
    record_keys: Sequence[str],
    name: str,
    source_path: Path,
) -> float | None:
    value = _first_present(payload, payload_keys)
    if value is None:
        value = _first_present(record, record_keys)
    if value is None:
        return None
    return _to_float(value, name=name, source_path=source_path)


def list_em_artifact_paths(
    artifact_dir: str,
    glob_pattern: str = '*.json',
    recursive: bool = False,
) -> list[Path]:
    """List EM artifact files from a directory."""

    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f'artifact_dir does not exist: {artifact_dir}')
    if not root.is_dir():
        raise NotADirectoryError(f'artifact_dir is not a directory: {artifact_dir}')

    iterator = root.rglob(glob_pattern) if recursive else root.glob(glob_pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No artifact files found in {artifact_dir} matching pattern '{glob_pattern}' "
            f'(recursive={recursive}).'
        )
    return paths


def load_em_run_summary(path: str | Path) -> EMRunSummary:
    """Load one EM artifact JSON into canonical run-level summary fields."""

    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'Expected JSON object in {source_path}.')

    payload_raw = raw.get('em_payload_for_boosting', {})
    payload: Mapping[str, object] = payload_raw if isinstance(payload_raw, dict) else {}

    target_margin_loss = _resolve_required_float(
        payload,
        raw,
        payload_keys=('target_margin_loss',),
        record_keys=('train_margin_loss',),
        name='target_margin_loss',
        source_path=source_path,
    )
    target_classification_error = _resolve_required_float(
        payload,
        raw,
        payload_keys=('target_classification_error',),
        record_keys=('train_error',),
        name='target_classification_error',
        source_path=source_path,
    )
    target_mean_margin = _resolve_required_float(
        payload,
        raw,
        payload_keys=('target_mean_margin',),
        record_keys=('train_margin_mean',),
        name='target_mean_margin',
        source_path=source_path,
    )
    reference_diagnostics = normalize_reference_diagnostics_v1(raw, source_path=source_path)
    reference_test_accuracy = reference_diagnostics.reference_test_accuracy
    reference_validation_accuracy = reference_diagnostics.reference_validation_accuracy

    margin_threshold = _resolve_optional_float(
        payload,
        raw,
        payload_keys=('margin_threshold', 'rho'),
        record_keys=('margin_threshold', 'rho'),
        default_value=float('nan'),
        name='margin_threshold',
        source_path=source_path,
    )
    rho = margin_threshold  # legacy alias retained for older callers/artifacts
    margin_tag_raw = _first_present(payload, ('margin_tag',))
    if margin_tag_raw is None:
        margin_tag_raw = _first_present(raw, ('margin_tag',))
    margin_tag = None if margin_tag_raw is None else str(margin_tag_raw)

    n_shots_eff_value = _first_present(payload, ('n_shots_eff',))
    if n_shots_eff_value is None:
        n_shots_eff_value = _first_present(raw, ('n_shots_eff', 'n_shots'))
    if n_shots_eff_value is None:
        raise KeyError(f'Missing n_shots_eff/n_shots in {source_path}.')
    n_shots_eff = _to_int(n_shots_eff_value, name='n_shots_eff', source_path=source_path)

    pec_gamma_sq = _resolve_optional_float(
        payload,
        raw,
        payload_keys=('pec_gamma_sq',),
        record_keys=('pec_gamma_sq',),
        default_value=1.0,
        name='pec_gamma_sq',
        source_path=source_path,
    )

    n_shots_raw_pec = _resolve_optional_float(
        payload,
        raw,
        payload_keys=('n_shots_raw_pec',),
        record_keys=('n_shots_raw_pec',),
        default_value=float(n_shots_eff) * float(pec_gamma_sq),
        name='n_shots_raw_pec',
        source_path=source_path,
    )

    run_id_raw = _first_present(raw, ('run_id', 'seed', 'train_key'))
    run_id = str(source_path.stem if run_id_raw is None else run_id_raw)

    n_qubits_raw = _first_present(raw, ('n_qubits',))
    n_layers_raw = _first_present(raw, ('n_layers',))
    n_train_raw = _first_present(raw, ('n_train_samples',))
    n_validation_raw = _first_present(raw, ('n_validation_samples',))
    n_test_raw = _first_present(raw, ('n_test_samples',))
    n_epochs_run_raw = _first_present(raw, ('n_epochs_run', 'epochs_run'))

    shot_accounting_raw = _first_present(
        payload,
        ('pec_shot_accounting', 'shot_accounting'),
    )
    if shot_accounting_raw is None:
        shot_accounting_raw = _first_present(
            raw,
            ('pec_shot_accounting', 'shot_accounting'),
        )
    shot_accounting = dict(shot_accounting_raw) if isinstance(shot_accounting_raw, Mapping) else None

    return EMRunSummary(
        run_id=run_id,
        source_path=str(source_path.resolve()),
        target_margin_loss=target_margin_loss,
        target_classification_error=target_classification_error,
        target_mean_margin=target_mean_margin,
        reference_test_accuracy=reference_test_accuracy,
        rho=rho,
        margin_threshold=margin_threshold,
        margin_tag=margin_tag,
        n_shots_eff=n_shots_eff,
        pec_gamma_sq=pec_gamma_sq,
        n_shots_raw_pec=n_shots_raw_pec,
        reference_validation_accuracy=reference_validation_accuracy,
        dataset_kind=str(raw.get('dataset_kind')) if raw.get('dataset_kind') is not None else None,
        n_qubits=int(n_qubits_raw) if n_qubits_raw is not None else None,
        n_layers=int(n_layers_raw) if n_layers_raw is not None else None,
        n_train_samples=int(n_train_raw) if n_train_raw is not None else None,
        n_validation_samples=int(n_validation_raw) if n_validation_raw is not None else None,
        n_test_samples=int(n_test_raw) if n_test_raw is not None else None,
        n_epochs_run=int(n_epochs_run_raw) if n_epochs_run_raw is not None else None,
        shot_accounting=shot_accounting,
    )


def load_em_run_summaries(paths: Iterable[str | Path]) -> list[EMRunSummary]:
    """Load multiple EM artifact files into canonical summaries."""

    summaries = [load_em_run_summary(path) for path in paths]
    if not summaries:
        raise ValueError('No EM artifact paths were provided.')
    return summaries


def load_em_run_summaries_from_dir(
    artifact_dir: str,
    glob_pattern: str = '*.json',
    recursive: bool = False,
) -> list[EMRunSummary]:
    """Discover and load EM artifact files from one directory."""

    return load_em_run_summaries(
        list_em_artifact_paths(
            artifact_dir=artifact_dir,
            glob_pattern=glob_pattern,
            recursive=recursive,
        )
    )


def _pessimistic_percentile(values: np.ndarray, percentile_q: float, lower_is_better: bool) -> float:
    q = percentile_q if lower_is_better else (1.0 - percentile_q)
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(values, q=q))


def _pessimistic_cvar(values: np.ndarray, tail_fraction: float, lower_is_better: bool) -> float:
    if values.size == 0:
        raise ValueError('values must be non-empty.')
    n_tail = max(1, int(math.ceil(float(tail_fraction) * values.size)))
    sorted_values = np.sort(values)
    tail = sorted_values[-n_tail:] if lower_is_better else sorted_values[:n_tail]
    return float(np.mean(tail))


def aggregate_em_run_targets(
    run_summaries: Sequence[EMRunSummary],
    target_name: str = 'margin_loss',
    percentile_q: float = 0.9,
    cvar_tail_fraction: float = 0.2,
    lower_is_better: bool | None = None,
) -> EMTargetAggregationSummary:
    """Aggregate independent EM runs into robust target statistics for boosting."""

    if not run_summaries:
        raise ValueError('run_summaries must be non-empty.')
    if target_name not in _TARGET_FIELD_AND_ORIENTATION:
        raise ValueError(f'target_name must be one of {sorted(_TARGET_FIELD_AND_ORIENTATION)}.')
    if not (0.0 < percentile_q < 1.0):
        raise ValueError('percentile_q must lie in (0, 1).')
    if not (0.0 < cvar_tail_fraction <= 1.0):
        raise ValueError('cvar_tail_fraction must lie in (0, 1].')

    target_field, default_lower_is_better = _TARGET_FIELD_AND_ORIENTATION[target_name]
    lower_is_better_effective = default_lower_is_better if lower_is_better is None else bool(lower_is_better)

    raw_values = [getattr(summary, target_field) for summary in run_summaries]
    if any(value is None for value in raw_values):
        raise ValueError(
            f"Target '{target_name}' is missing from at least one EM summary. "
            'Make sure validation-aware EM artifacts were generated first.'
        )
    values = np.asarray(raw_values, dtype=np.float64)

    pathwise_pessimistic = float(np.max(values) if lower_is_better_effective else np.min(values))
    percentile_value = _pessimistic_percentile(
        values,
        percentile_q=percentile_q,
        lower_is_better=lower_is_better_effective,
    )
    cvar_value = _pessimistic_cvar(
        values,
        tail_fraction=cvar_tail_fraction,
        lower_is_better=lower_is_better_effective,
    )

    return EMTargetAggregationSummary(
        target_name=target_name,
        n_runs=int(values.size),
        lower_is_better=lower_is_better_effective,
        pathwise_pessimistic=pathwise_pessimistic,
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        percentile=percentile_value,
        percentile_q=float(percentile_q),
        cvar=cvar_value,
        cvar_tail_fraction=float(cvar_tail_fraction),
    )


def resolve_em_target_from_aggregation(
    summary: EMTargetAggregationSummary,
    strategy: str = 'pathwise_pessimistic',
) -> float:
    """Pick one scalar target from aggregated EM statistics."""

    valid = {'pathwise_pessimistic', 'mean', 'median', 'percentile', 'cvar'}
    if strategy not in valid:
        raise ValueError(f'strategy must be one of {sorted(valid)}.')

    return {
        'pathwise_pessimistic': float(summary.pathwise_pessimistic),
        'mean': float(summary.mean),
        'median': float(summary.median),
        'percentile': float(summary.percentile),
        'cvar': float(summary.cvar),
    }[strategy]


def build_boosting_target_candidates(summary: EMTargetAggregationSummary) -> dict[str, float]:
    """Return all available scalar target candidates for downstream boosting runs."""

    return {
        'pathwise_pessimistic': float(summary.pathwise_pessimistic),
        'mean': float(summary.mean),
        'median': float(summary.median),
        'percentile': float(summary.percentile),
        'cvar': float(summary.cvar),
    }


def build_boosting_stop_target(
    *,
    target_margin_loss: float,
    target_validation_accuracy: float,
    validation_accuracy_tolerance: float = 0.02,
    min_rounds: int = 1,
    patience: int = 1,
) -> BoostingStopTargetConfig:
    """Construct a validation-aware boosting stop target from EM summary statistics."""

    return BoostingStopTargetConfig(
        target_margin_loss=float(target_margin_loss),
        target_validation_accuracy=float(target_validation_accuracy),
        validation_accuracy_tolerance=float(validation_accuracy_tolerance),
        min_rounds=int(min_rounds),
        patience=int(patience),
    )


def build_boosting_stop_target_from_summary(
    summary: EMRunSummary,
    *,
    validation_accuracy_tolerance: float = 0.02,
    min_rounds: int = 1,
    patience: int = 1,
) -> BoostingStopTargetConfig:
    """Build a stop target directly from one EM run summary."""

    if summary.reference_validation_accuracy is None or not np.isfinite(summary.reference_validation_accuracy):
        raise ValueError(
            'EM summary does not contain reference_validation_accuracy. '
            'Generate EM artifacts with a validation split first.'
        )
    return build_boosting_stop_target(
        target_margin_loss=float(summary.target_margin_loss),
        target_validation_accuracy=float(summary.reference_validation_accuracy),
        validation_accuracy_tolerance=validation_accuracy_tolerance,
        min_rounds=min_rounds,
        patience=patience,
    )



def build_boosting_stop_target_from_run_summaries(
    run_summaries: Sequence[EMRunSummary],
    *,
    margin_strategy: str = 'pathwise_pessimistic',
    validation_strategy: str = 'pathwise_pessimistic',
    percentile_q: float = 0.9,
    cvar_tail_fraction: float = 0.2,
    validation_accuracy_tolerance: float = 0.02,
    min_rounds: int = 1,
    patience: int = 1,
) -> tuple[BoostingStopTargetConfig, EMTargetAggregationSummary, EMTargetAggregationSummary]:
    """Aggregate EM runs and directly construct a validation-aware boosting stop target."""

    margin_summary = aggregate_em_run_targets(
        run_summaries,
        target_name='margin_loss',
        percentile_q=percentile_q,
        cvar_tail_fraction=cvar_tail_fraction,
    )
    validation_summary = aggregate_em_run_targets(
        run_summaries,
        target_name='validation_accuracy',
        percentile_q=percentile_q,
        cvar_tail_fraction=cvar_tail_fraction,
    )
    stop_target = build_boosting_stop_target(
        target_margin_loss=resolve_em_target_from_aggregation(margin_summary, strategy=margin_strategy),
        target_validation_accuracy=resolve_em_target_from_aggregation(
            validation_summary,
            strategy=validation_strategy,
        ),
        validation_accuracy_tolerance=validation_accuracy_tolerance,
        min_rounds=min_rounds,
        patience=patience,
    )
    return stop_target, margin_summary, validation_summary

def load_and_aggregate_em_runs(
    artifact_config: EMRunArtifactConfig,
    aggregation_config: EMTargetAggregationConfig,
) -> tuple[list[EMRunSummary], EMTargetAggregationSummary]:
    """End-to-end helper: discover run files, load them, and aggregate targets."""

    run_summaries = load_em_run_summaries_from_dir(
        artifact_dir=artifact_config.artifact_dir,
        glob_pattern=artifact_config.glob_pattern,
        recursive=artifact_config.recursive,
    )
    aggregate = aggregate_em_run_targets(
        run_summaries,
        target_name=aggregation_config.target_name,
        percentile_q=aggregation_config.percentile_q,
        cvar_tail_fraction=aggregation_config.cvar_tail_fraction,
        lower_is_better=aggregation_config.lower_is_better,
    )
    return run_summaries, aggregate
