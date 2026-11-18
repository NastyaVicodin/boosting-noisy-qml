"""Versioned normalization of reference diagnostic fields in EM artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping


CANONICAL_REFERENCE_DIAGNOSTICS_SCHEMA = "boosting_qml.reference_diagnostics.v1"
STRICT_V5_STAGE_B_SCHEMA = "strict_v5.iid_high_margin.prospective_reference_run.v1"
STRICT_V5_FINAL_DEPTH_REFERENCE_SCHEMA = "strict_v5.final_depth_expansion.reference_run.v1"


@dataclass(frozen=True)
class CanonicalReferenceDiagnostics:
    """Canonical diagnostic metadata separated from scientific EM inputs."""

    schema_version: str
    source_schema_version: str | None
    reference_test_accuracy: float | None
    reference_validation_accuracy: float | None
    test_source: str
    validation_source: str
    test_status: str


def _finite_float(value: object, *, name: str, source_path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse float for '{name}' from {source_path}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite '{name}' in {source_path}, got {result!r}.")
    return result


def _equivalent_optional_float(
    payload: Mapping[str, object],
    record: Mapping[str, object],
    *,
    payload_key: str,
    record_key: str,
    canonical_name: str,
    source_path: Path,
) -> tuple[float | None, str]:
    payload_value = payload.get(payload_key)
    record_value = record.get(record_key)
    parsed: list[tuple[str, float]] = []
    if payload_value is not None:
        parsed.append(
            (
                f"em_payload_for_boosting.{payload_key}",
                _finite_float(payload_value, name=canonical_name, source_path=source_path),
            )
        )
    if record_value is not None:
        parsed.append(
            (record_key, _finite_float(record_value, name=canonical_name, source_path=source_path))
        )
    if len(parsed) == 2 and not math.isclose(parsed[0][1], parsed[1][1], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Conflicting equivalent fields for '{canonical_name}' in {source_path}: "
            f"{parsed[0][0]}={parsed[0][1]!r}, {parsed[1][0]}={parsed[1][1]!r}."
        )
    if not parsed:
        return None, "absent"
    return parsed[0][1], "+".join(name for name, _ in parsed)


def _strict_v5_censored_without_test(record: Mapping[str, object]) -> bool:
    stage_b = record.get("strict_v5_stage_b")
    if not isinstance(stage_b, Mapping):
        stage_b = record.get("strict_v5_final_depth_reference")
    target = record.get("prospective_train_margin_target")
    if not isinstance(stage_b, Mapping) or not isinstance(target, Mapping):
        return False
    if stage_b.get("schema_version") not in {
        STRICT_V5_STAGE_B_SCHEMA,
        STRICT_V5_FINAL_DEPTH_REFERENCE_SCHEMA,
    }:
        return False
    selected_metrics = stage_b.get("deterministic_selected_metrics")
    return (
        target.get("target_reached") is False
        and target.get("test_evaluated_once") is False
        and isinstance(selected_metrics, Mapping)
        and len(selected_metrics) == 0
    )


def normalize_reference_diagnostics_v1(
    record: Mapping[str, object],
    *,
    source_path: str | Path,
) -> CanonicalReferenceDiagnostics:
    """Normalize equivalent diagnostic aliases without inventing missing values.

    A null test diagnostic is accepted only for the explicitly recognized
    strict-v5 Stage-B right-censored state, where no selected predictor was
    test-evaluated. Other missing or conflicting test fields fail loudly.
    """

    resolved_path = Path(source_path)
    payload_raw = record.get("em_payload_for_boosting", {})
    payload: Mapping[str, object] = payload_raw if isinstance(payload_raw, Mapping) else {}
    stage_b = record.get("strict_v5_stage_b")
    if not isinstance(stage_b, Mapping):
        stage_b = record.get("strict_v5_final_depth_reference")
    source_schema = (
        str(stage_b.get("schema_version"))
        if isinstance(stage_b, Mapping) and stage_b.get("schema_version") is not None
        else None
    )

    test_accuracy, test_source = _equivalent_optional_float(
        payload,
        record,
        payload_key="reference_test_accuracy",
        record_key="test_accuracy",
        canonical_name="reference_test_accuracy",
        source_path=resolved_path,
    )
    validation_accuracy, validation_source = _equivalent_optional_float(
        payload,
        record,
        payload_key="reference_validation_accuracy",
        record_key="validation_accuracy",
        canonical_name="reference_validation_accuracy",
        source_path=resolved_path,
    )

    censored_without_test = _strict_v5_censored_without_test(record)
    if test_accuracy is None and not censored_without_test:
        raise KeyError(
            f"Missing non-null reference test diagnostic in {resolved_path}. "
            "Canonical null is permitted only for an explicitly recognized strict-v5 "
            "Stage-B right-censored record with test_evaluated_once=false."
        )
    if test_accuracy is not None and censored_without_test:
        raise ValueError(
            f"Contradictory strict-v5 Stage-B censored record in {resolved_path}: "
            "a test diagnostic is present although test_evaluated_once=false."
        )

    return CanonicalReferenceDiagnostics(
        schema_version=CANONICAL_REFERENCE_DIAGNOSTICS_SCHEMA,
        source_schema_version=source_schema,
        reference_test_accuracy=test_accuracy,
        reference_validation_accuracy=validation_accuracy,
        test_source=test_source,
        validation_source=validation_source,
        test_status=(
            "not_evaluated_right_censored"
            if censored_without_test
            else "selected_predictor_finite_shot_diagnostic"
        ),
    )
