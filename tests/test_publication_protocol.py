"""Small regressions against the copied frozen execution helpers.

These tests use toy scalar records and do not evaluate a circuit or read any
held-out test array. The full archive checks live in scripts/verify_archive.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "base" / "source"))

from audit import Ledger  # noqa: E402
from endpoints import EndpointTracker, algorithm_shots, classification_metrics  # noqa: E402


spec = importlib.util.spec_from_file_location(
    "frozen_reference_patience", ROOT / "experiments/reference/source/patience.py")
assert spec and spec.loader
patience = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patience)

verifier_spec = importlib.util.spec_from_file_location(
    "publication_verifier", ROOT / "scripts/verify_archive.py")
assert verifier_spec and verifier_spec.loader
verifier = importlib.util.module_from_spec(verifier_spec)
verifier_spec.loader.exec_module(verifier)


def test_archive_scan_excludes_only_root_git_metadata(tmp_path: Path) -> None:
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "example").write_text("git metadata")
    (tmp_path / "extra" / ".git").mkdir(parents=True)
    ordinary = tmp_path / "extra" / "unexpected.txt"
    ordinary.write_text("must remain visible")
    linked = tmp_path / "extra" / "linked.txt"
    linked.symlink_to(ordinary)

    entries = set(verifier.snapshot_entries(tmp_path))
    assert tmp_path / ".git" not in entries
    assert tmp_path / ".git" / "objects" / "example" not in entries
    assert ordinary in entries
    assert tmp_path / "extra" / ".git" in entries
    assert linked in entries


def test_patience_is_strict_and_counts_all_hundred_stale_epochs() -> None:
    tracker = patience.Patience100()
    assert tracker.update(1.0, 1)["improved"]
    for epoch in range(2, 101):
        decision = tracker.update(1.0, epoch)
        assert not decision["improved"]
        assert not decision["should_stop"]
    decision = tracker.update(1.0, 101)
    assert decision["should_stop"]
    assert decision["best_epoch"] == 1
    assert decision["epochs_without_improvement"] == 100


def test_reference_epoch_statistic_is_arithmetic_mean_of_24_losses() -> None:
    records = [{"update": index, "loss": 1.0 if index < 24 else 25.0}
               for index in range(1, 25)]
    history = patience.scan_updates(records)
    assert history[0]["epoch_loss"] == 2.0


def test_algorithm_shots_and_first_hit_are_independent_of_confirmation(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, budget=100_000)
    for role, n, shots in (("fit", 4, 25), ("edge", 8, 25),
                           ("monitor", 3, 25), ("confirm", 3, 25),
                           ("diagnostic", 2, 25)):
        event = ledger.begin(role, n, shots)
        ledger.finish(event)
    assert algorithm_shots(ledger) == 300
    tracker = EndpointTracker(tmp_path, ledger)
    evaluations = []

    def evaluate(role: str) -> dict:
        evaluations.append(role)
        return {"passes": True}

    first = tracker.check("margin", {"round": 1}, {"prefix": 1}, evaluate)
    second = tracker.check("margin", {"round": 2}, {"prefix": 2}, evaluate)
    assert first is second
    assert first["algorithm_raw_shots"] == 300
    assert evaluations == ["monitor", "confirm"]
    assert (tmp_path / "first_hit_margin.json").is_file()


def test_executed_classification_cutoff_and_zero_score_tie() -> None:
    record = classification_metrics(
        np.array([0.0, -0.1, 0.2]), np.array([1, -1, -1]), 2)
    assert record["correct"] == 2
    assert record["passes"]
    assert record["executed_score_ties"] == 1
