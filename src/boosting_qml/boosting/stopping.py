"""Shared, auditable stopping-policy decisions for boosting."""

from __future__ import annotations

from typing import Any, Mapping

from boosting_qml.config import BoostingStopTargetConfig


def fixed_train_margin_loss_decision(
    round_result: Any,
    stop_target: BoostingStopTargetConfig,
    *,
    round_index: int,
) -> bool:
    """Return the margin-only stop decision for an accepted prefix.

    Validation and test fields on ``round_result`` are intentionally neither
    read nor accepted as inputs to the criterion.  The recorded online loss
    must already use the canonical normalized-ensemble ramp definition.
    """

    if stop_target.stop_mode != "fixed_train_margin_loss":
        raise ValueError("fixed_train_margin_loss_decision requires its matching stop mode.")
    if stop_target.train_margin_mu is None or stop_target.train_margin_target is None:
        raise ValueError("The fixed training-margin target is incomplete.")
    loss = float(round_result.empirical_train_margin_loss)
    return bool(
        int(round_index) >= int(stop_target.min_rounds)
        and loss <= float(stop_target.train_margin_target)
    )


def checkpoint_stop_target_matches(
    observed: Any,
    expected: dict[str, Any] | None,
) -> bool:
    """Compare persisted targets while accepting pre-extension legacy fields.

    Checkpoints written before ``fixed_train_margin_loss`` existed lack the
    three append-only mode fields.  They may resume only into a non-margin
    mode; a legacy checkpoint can never be inferred to be a margin run.
    """

    if observed is None or expected is None:
        return observed is expected
    if not isinstance(observed, Mapping):
        return False
    normalized = dict(observed)
    if "stop_mode" not in normalized:
        if expected.get("stop_mode") == "fixed_train_margin_loss":
            return False
        normalized["stop_mode"] = expected.get("stop_mode")
        normalized["train_margin_mu"] = None
        normalized["train_margin_target"] = None
    return normalized == expected
