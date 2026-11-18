#!/usr/bin/env python3
"""Tiny fresh PQC fit and scalar analysis; never uses paper test data.

This is an installation/behaviour smoke test. It is intentionally a new toy
case, not a rerun of a selected N=25 paper trajectory or a fidelity claim.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from boosting_qml.data.datasets import make_toy_binary_classification  # noqa: E402
from boosting_qml.weak_learners.pqc_binary import (  # noqa: E402
    WeightedPQCBinaryWeakLearner,
)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    data = make_toy_binary_classification(n_samples=8, n_features=4, seed=41)
    learner = WeightedPQCBinaryWeakLearner(
        n_qubits=4, n_layers=1, n_shots_eff=25, n_epochs=1,
        batch_size=4, noise_backend="density_matrix",
        noise_strength_1q=0.005, noise_strength_2q=0.005,
        measurement_sampling_distribution="binomial",
        measurement_gradient_mode="straight_through_expectation",
        compute_fit_diagnostic=False,
    )
    fitted = learner.fit_weighted(
        data, np.full(data.n_samples, 1.0 / data.n_samples),
        np.random.default_rng(42),
    )
    scores = np.asarray(learner.predict_score(data.X, rng=np.random.default_rng(43)))
    if scores.shape != (data.n_samples,) or not np.all(np.isfinite(scores)):
        raise AssertionError("Nonfinite or misshaped PQC prediction")
    margins = np.asarray(data.y, dtype=np.float64) * scores
    ramp = float(np.mean(np.clip(1.0 - margins / 0.1, 0.0, 1.0)))
    edge = float(np.mean(margins))
    if not math.isfinite(ramp) or not -1.0 <= edge <= 1.0:
        raise AssertionError("Invalid scalar analysis")
    report = {
        "scope": "new 8-example, one-epoch toy fit; not a paper rerun",
        "training_examples": data.n_samples,
        "epochs_run": int(fitted.epochs_run),
        "shots_per_forward_prediction": 25,
        "sampled_training_scores": scores.tolist(),
        "empirical_ramp_loss_mu_0p1": ramp,
        "uniform_weighted_edge": edge,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "optax": optax.__version__,
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
        "jax_default_backend": jax.default_backend(),
    }
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory for the smoke report")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"passed": True, "epochs_run": result["epochs_run"],
                      "output": str(args.output / "smoke_report.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
