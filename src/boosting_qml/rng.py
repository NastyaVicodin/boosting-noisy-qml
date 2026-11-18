"""Stable RNG seed helpers for experiment-level common random numbers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


INT32_MAX = int(np.iinfo(np.int32).max)
BOOSTING_CRN_NAMESPACE = "boosting_common_random_numbers"


def stable_int_seed(*parts: Any, modulo: int = INT32_MAX) -> int:
    """Return a deterministic positive int seed from semantic key parts."""
    if modulo <= 1:
        raise ValueError("modulo must be greater than 1.")
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(modulo - 1) + 1


def boosting_round_rng_seeds(base_seed: int, round_idx: int) -> dict[str, int]:
    """Per-round common-random-number seeds shared across boosting algorithms."""
    base = (BOOSTING_CRN_NAMESPACE, int(base_seed), "round", int(round_idx))
    return {
        "round": stable_int_seed(*base),
        "fit": stable_int_seed(*base, "fit"),
        "edge_score": stable_int_seed(*base, "edge_score"),
        "validation_monitor": stable_int_seed(*base, "validation_monitor"),
        "validation_confirm": stable_int_seed(*base, "validation_confirm"),
        "test_score": stable_int_seed(*base, "test_score"),
    }


def weighted_pqc_fit_seed(base_seed: int, *parts: Any) -> int:
    """Deterministic weak-PQC fit subkey for epoch/batch-level randomness."""
    return stable_int_seed("weighted_pqc_fit", int(base_seed), *parts)
