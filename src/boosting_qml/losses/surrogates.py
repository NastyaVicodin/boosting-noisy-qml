"""Smooth surrogate losses used by EM-style PQC training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jaxtyping import Float

from boosting_qml.eval.metrics import Batch, get_metrics
from boosting_qml.pqc.models import predict


def standard_normal_cdf(v):
    return 0.5 * (1 + erf(v / jnp.sqrt(2)))


def standard_normal_pdf(v):
    return (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-v**2 / 2)


def PsiG(alpha, sigma=0.5):
    v = (1.0 - alpha) / sigma
    return standard_normal_cdf(v) * (1.0 - alpha) + standard_normal_pdf(v) * sigma


def smooth_hinge_loss_PsiG(
    model: Any,
    batch: Batch,
    key_shot_noise: jax.random.PRNGKey,
    sigma: Float = 0.5,
):
    ypred = predict(model, batch.inputs, key_shot_noise)
    alpha = batch.labels * ypred
    per_example_loss = PsiG(alpha, sigma)
    loss = jnp.mean(per_example_loss)
    metrics = get_metrics(ypred, batch)
    return loss, metrics


@jax.jit
def empirical_hinge_margin_loss(y_true: jax.Array, y_pred: jax.Array, margin_threshold: Float) -> jax.Array:
    def _margin_loss_step(_, inputs):
        y_t, y_p = inputs
        margin = y_t * y_p
        loss = jnp.where(
            margin >= margin_threshold,
            0.0,
            jnp.where(
                margin >= 0,
                1.0 - margin / margin_threshold,
                1.0,
            ),
        )
        return _, loss

    _, losses = jax.lax.scan(_margin_loss_step, None, (y_true, y_pred))
    return jnp.mean(losses)


@jax.jit
def classification_error_from_scores(
    y_true: jax.Array,
    y_pred: jax.Array,
    threshold: Float = 0.0,
) -> jax.Array:
    pred = 2 * (y_pred >= threshold).astype(jnp.int32) - 1
    y_true_int = y_true.astype(jnp.int32)
    return jnp.mean(pred != y_true_int)


__all__ = [
    "PsiG",
    "classification_error_from_scores",
    "empirical_hinge_margin_loss",
    "smooth_hinge_loss_PsiG",
    "standard_normal_cdf",
    "standard_normal_pdf",
]
