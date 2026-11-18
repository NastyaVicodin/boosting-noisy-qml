"""Simple classical real-valued weak learner for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from boosting_qml.types import Array, BinaryDataset
from boosting_qml.weak_learners.base import BinaryWeakLearner, WeakLearnerFitResult

_EPS = np.finfo(np.float64).eps


@dataclass
class WeightedLinearWeakLearner(BinaryWeakLearner):
    """Weighted ridge regressor squashed with tanh to produce scores in [-1, 1]."""

    ridge_lambda: float = 1e-2
    score_scale: float = 1.0
    fit_objective: str = "weighted_mse"
    edge_learning_rate: float = 0.05
    edge_n_steps: int = 300
    edge_grad_clip_norm: float = 5.0
    coef_: Optional[Array] = field(default=None, init=False, repr=False)
    intercept_: float = field(default=0.0, init=False)

    def fit_weighted(
        self,
        dataset: BinaryDataset,
        example_weights: Array,
        rng: np.random.Generator,
    ) -> WeakLearnerFitResult:
        del rng  # deterministic fit

        x = np.asarray(dataset.X, dtype=np.float64)
        y = np.asarray(dataset.y, dtype=np.float64)
        w = np.asarray(example_weights, dtype=np.float64)

        if w.shape != (dataset.n_samples,):
            raise ValueError("example_weights must have shape [n_samples].")
        if self.fit_objective not in {"weighted_mse", "edge_max"}:
            raise ValueError("fit_objective must be 'weighted_mse' or 'edge_max'.")

        w = np.clip(w, _EPS, None)
        w = w / np.sum(w)

        if self.fit_objective == "weighted_mse":
            self._fit_weighted_mse(x=x, y=y, w=w)
            pred = self.predict_score(x)
            weighted_mse = float(np.average((pred - y) ** 2, weights=w))
            return WeakLearnerFitResult(weighted_surrogate_loss_t=weighted_mse)

        self._fit_edge_max(x=x, y=y, w=w)
        pred = self.predict_score(x)
        edge = float(np.dot(w, y * pred))
        reg = 0.5 * self.ridge_lambda * float(np.dot(self.coef_, self.coef_)) if self.coef_ is not None else 0.0
        # Surrogate tracked as a minimization value; smaller is better.
        objective_as_loss = float(-(edge - reg))
        return WeakLearnerFitResult(weighted_surrogate_loss_t=objective_as_loss)

    def _fit_weighted_mse(self, x: Array, y: Array, w: Array) -> None:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        sqrt_w = np.sqrt(w)[:, None]

        xw = x_aug * sqrt_w
        yw = y * sqrt_w[:, 0]

        n_features_aug = x_aug.shape[1]
        reg = np.eye(n_features_aug, dtype=np.float64) * self.ridge_lambda
        reg[-1, -1] = 0.0  # do not regularize bias

        beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
        self.coef_ = beta[:-1]
        self.intercept_ = float(beta[-1])

    def _fit_edge_max(self, x: Array, y: Array, w: Array) -> None:
        # Initialize from weighted MSE for faster convergence.
        self._fit_weighted_mse(x=x, y=y, w=w)
        assert self.coef_ is not None

        coef = self.coef_.copy()
        intercept = float(self.intercept_)
        lr = float(self.edge_learning_rate)
        grad_clip = float(self.edge_grad_clip_norm)

        for _ in range(int(self.edge_n_steps)):
            logits = self.score_scale * (x @ coef + intercept)
            scores = np.tanh(logits)
            deriv = 1.0 - scores**2
            common = w * y * deriv * self.score_scale

            grad_coef = x.T @ common - self.ridge_lambda * coef
            grad_intercept = float(np.sum(common))

            grad_norm = float(np.sqrt(np.sum(grad_coef**2) + grad_intercept**2))
            if grad_norm > grad_clip and grad_norm > 0.0:
                scale = grad_clip / grad_norm
                grad_coef *= scale
                grad_intercept *= scale

            coef = coef + lr * grad_coef
            intercept = intercept + lr * grad_intercept

        self.coef_ = coef
        self.intercept_ = intercept

    def predict_score(
        self,
        x: Array,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        del rng
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit_weighted first.")

        x = np.asarray(x, dtype=np.float64)
        raw = x @ self.coef_ + self.intercept_
        score = np.tanh(self.score_scale * raw)
        return np.clip(score, -1.0, 1.0)

    def predict_label(
        self,
        x: Array,
        threshold: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        score = self.predict_score(x, rng=rng)
        return np.where(score >= threshold, 1, -1).astype(np.int8)

    def state_dict(self) -> Mapping[str, Any]:
        if self.coef_ is None:
            raise RuntimeError("Cannot serialize an unfitted weak learner.")
        return {
            "weak_learner_type": "weighted_linear",
            "ridge_lambda": self.ridge_lambda,
            "score_scale": self.score_scale,
            "fit_objective": self.fit_objective,
            "edge_learning_rate": self.edge_learning_rate,
            "edge_n_steps": self.edge_n_steps,
            "edge_grad_clip_norm": self.edge_grad_clip_norm,
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "WeightedLinearWeakLearner":
        learner = cls(
            ridge_lambda=float(state["ridge_lambda"]),
            score_scale=float(state["score_scale"]),
            fit_objective=str(state.get("fit_objective", "weighted_mse")),
            edge_learning_rate=float(state.get("edge_learning_rate", 0.05)),
            edge_n_steps=int(state.get("edge_n_steps", 300)),
            edge_grad_clip_norm=float(state.get("edge_grad_clip_norm", 5.0)),
        )
        learner.coef_ = np.asarray(state["coef"], dtype=np.float64)
        learner.intercept_ = float(state["intercept"])
        return learner
