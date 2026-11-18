"""Deterministic helpers for weak-learner versus ensemble diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ROUND_COUNT_BINS = (
    "exactly_1",
    "2_to_4",
    "5_to_10",
    "11_to_25",
    "26_to_50",
    "51_to_100",
    "101_to_499",
    "500",
)


def strict_margin_accuracy(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Ordinary unweighted accuracy using the requested strict ``y*g > 0``."""

    y = np.asarray(labels, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if y.shape != values.shape or y.ndim != 1:
        raise ValueError("labels and scores must be equal-length vectors")
    if not np.all(np.isin(y, (-1.0, 1.0))) or not np.all(np.isfinite(values)):
        raise ValueError("labels must be +/-1 and scores finite")
    return float(np.mean(y * values > 0.0))


def production_tie_positive_accuracy(
    labels: Sequence[float], scores: Sequence[float]
) -> float:
    """Production accuracy where a zero score predicts the positive class."""

    y = np.asarray(labels, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if y.shape != values.shape or y.ndim != 1:
        raise ValueError("labels and scores must be equal-length vectors")
    return float(np.mean(np.where(values >= 0.0, 1.0, -1.0) == y))


def round_count_bin(accepted_rounds: int) -> str:
    rounds = int(accepted_rounds)
    if rounds < 1 or rounds > 500:
        raise ValueError("accepted_rounds must lie in [1,500]")
    if rounds == 1:
        return "exactly_1"
    if rounds <= 4:
        return "2_to_4"
    if rounds <= 10:
        return "5_to_10"
    if rounds <= 25:
        return "11_to_25"
    if rounds <= 50:
        return "26_to_50"
    if rounds <= 100:
        return "51_to_100"
    if rounds <= 499:
        return "101_to_499"
    return "500"


def hit_by_k_status(
    *, first_hit_round: int | None, accepted_rounds: int, k: int
) -> str:
    """Observed hit, observed non-hit, or censored-before-K classification."""

    accepted = int(accepted_rounds)
    cutoff = int(k)
    if cutoff < 1 or accepted < 1:
        raise ValueError("k and accepted_rounds must be positive")
    if first_hit_round is not None and int(first_hit_round) <= cutoff:
        return "observed_hit_by_K"
    if accepted >= cutoff:
        return "observed_non_hit_by_K"
    return "not_observable_censored_before_K"


def representative_trajectory(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Predeclared median-round representative with lexicographic tie-break."""

    if not rows:
        raise ValueError("representative selection requires nonempty rows")
    median_rounds = float(np.median([int(row["accepted_rounds"]) for row in rows]))
    return min(
        rows,
        key=lambda row: (
            abs(int(row["accepted_rounds"]) - median_rounds),
            int(row["depth"]),
            float(row["noise_strength"]),
            int(row["task_index"]),
            int(row["run_index"]),
        ),
    )


__all__ = [
    "ROUND_COUNT_BINS",
    "hit_by_k_status",
    "production_tie_positive_accuracy",
    "representative_trajectory",
    "round_count_bin",
    "strict_margin_accuracy",
]
