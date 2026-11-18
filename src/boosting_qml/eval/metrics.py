"""Evaluation structs and binary metrics."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxtyping import Float, Int


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


class Result(NamedTuple):
    tp: Int
    tn: Int
    fp: Int
    fn: Int
    accuracy: Float
    precision: Float
    recall: Float
    fscore: Float


def get_accuracy(
    pred: jax.Array,
    actual: jax.Array,
    eps: Float = jnp.finfo(float).eps,
) -> Tuple[Float, Int, Int, Int, Int]:
    tp: int = jnp.sum(jnp.logical_and(pred == 1, actual == 1))
    tn: int = jnp.sum(jnp.logical_and(pred == -1, actual == -1))
    fp: int = jnp.sum(jnp.logical_and(pred == 1, actual == -1))
    fn: int = jnp.sum(jnp.logical_and(pred == -1, actual == 1))
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return accuracy, tp, tn, fp, fn


def get_metrics(
    pred: jax.Array,
    batch: Batch,
    threshold: Float = 0.0,
    eps: Float = jnp.finfo(float).eps,
) -> Result:
    actual = batch.labels
    pred = 2 * (pred >= threshold).astype(jnp.int32) - 1

    accuracy, tp, tn, fp, fn = get_accuracy(pred, actual)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)

    return Result(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )


__all__ = ["Batch", "Result", "get_accuracy", "get_metrics"]
