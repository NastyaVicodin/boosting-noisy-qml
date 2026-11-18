"""Training-step helpers for EM-style PQC experiments."""

from __future__ import annotations

from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from jaxtyping import Float, Int, PyTree

from boosting_qml.eval.metrics import Batch, Result, get_metrics
from boosting_qml.losses.surrogates import smooth_hinge_loss_PsiG
from boosting_qml.pqc.models import ErrorMitigatedPQCModel
from boosting_qml.training.finite_shot_parameter_shift import (
    finite_shot_plugin_loss_grads_and_preds_density,
    legacy_shifted_gate_occurrence_count,
)

@dataclass
class StaticConfig:
    n_qubits: Int = eqx.field(static=True)
    n_layers: Int = eqx.field(static=True)
    n_shots: Int = eqx.field(static=True)
    batch_size: Int = eqx.field(static=True)
    n_minibatches: Int = eqx.field(static=True)
    grad_clip_norm: Float = eqx.field(static=True)
    learning_rate: Float = eqx.field(static=True)
    n_epochs: Int = eqx.field(static=True)
    train_key: Int = eqx.field(static=True)
    test_key: Int = eqx.field(static=True)
    margin_threshold: Float = eqx.field(static=True)


def accumulate_gradients_scan(
    model: ErrorMitigatedPQCModel,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: Int,
    loss_fn: Callable = smooth_hinge_loss_PsiG,
) -> Tuple[PyTree, float, Result]:
    batch_size = batch.inputs.shape[0]
    if batch_size % int(num_minibatches) != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})."
        )
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, num_minibatches)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def _minibatch_step(minibatch_idx: jax.Array | int) -> Tuple[PyTree, float, Result]:
        minibatch = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, start_index=minibatch_idx * minibatch_size, slice_size=minibatch_size, axis=0
            ),
            batch,
        )
        (step_loss, step_metrics), step_grads = grad_fn(
            model, minibatch, rngs[minibatch_idx]
        )
        return step_grads, step_loss, step_metrics

    def _scan_step(
        carry: Tuple[PyTree, float, Result], minibatch_idx: jax.Array | int
    ) -> Tuple[Tuple[PyTree, float, Result], None]:
        step_grads, step_loss, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree_util.tree_map(jnp.add, carry, (step_grads, step_loss, step_metrics))
        return carry, None

    grads, loss, metrics = _minibatch_step(0)
    if int(num_minibatches) > 1:
        (grads, loss, metrics), _ = jax.lax.scan(
            _scan_step,
            init=(grads, loss, metrics),
            xs=jnp.arange(1, num_minibatches),
        )

    grads = jax.tree_util.tree_map(lambda g: g / num_minibatches, grads)
    loss = loss / num_minibatches
    metrics = jax.tree_util.tree_map(lambda m: m / num_minibatches, metrics)
    return grads, loss, metrics


def _legacy_shifted_gate_occurrence_count(model: ErrorMitigatedPQCModel) -> int:
    return legacy_shifted_gate_occurrence_count(
        n_qubits=int(model.n_qubits),
        n_layers=int(model.n_layers),
    )


def _finite_shot_plugin_loss_and_grad_params(
    model: ErrorMitigatedPQCModel,
    batch: Batch,
    rng: jax.random.PRNGKey,
    *,
    sigma: Float = 0.5,
) -> tuple[jax.Array, jax.Array, Result]:
    grads_params, loss, preds0 = finite_shot_plugin_loss_grads_and_preds_density(
        model.get_params(),
        batch.inputs,
        batch.labels,
        rng,
        n_qubits=int(model.n_qubits),
        n_layers=int(model.n_layers),
        n_shots=int(model.n_shots),
        noise_strength_1q=float(model.noise_strength_1q),
        noise_strength_2q=float(model.noise_strength_2q),
        apply_two_qubit_noise=bool(model.mc_apply_two_qubit_noise),
        weights=None,
        sigma=float(sigma),
        ansatz_kind=str(model.ansatz_kind),
        noise_backend=str(model.noise_backend),
        measurement_sampling_distribution=str(model.measurement_sampling_distribution),
        measurement_gradient_mode=str(model.measurement_gradient_mode),
        observable="global_z",
        parameter_shift_chunk_size=int(model.parameter_shift_chunk_size),
    )
    metrics = get_metrics(preds0, batch)
    return grads_params, loss, metrics


def accumulate_parameter_shift_gradients_scan(
    model: ErrorMitigatedPQCModel,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: Int,
) -> Tuple[jax.Array, float, Result]:
    batch_size = batch.inputs.shape[0]
    if batch_size % int(num_minibatches) != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})."
        )
    minibatch_size = batch_size // num_minibatches
    rngs = jax.random.split(rng, int(num_minibatches))

    grads_total = None
    loss_total = jnp.asarray(0.0, dtype=batch.inputs.dtype)
    metrics_total = None
    for minibatch_idx in range(int(num_minibatches)):
        start = minibatch_idx * minibatch_size
        stop = start + minibatch_size
        minibatch = Batch(
            inputs=batch.inputs[start:stop],
            labels=batch.labels[start:stop],
        )
        step_grads, step_loss, step_metrics = _finite_shot_plugin_loss_and_grad_params(
            model,
            minibatch,
            rngs[minibatch_idx],
        )
        grads_total = step_grads if grads_total is None else grads_total + step_grads
        loss_total = loss_total + step_loss
        metrics_total = (
            step_metrics
            if metrics_total is None
            else jax.tree_util.tree_map(jnp.add, metrics_total, step_metrics)
        )

    grads = grads_total / int(num_minibatches)
    loss = loss_total / int(num_minibatches)
    metrics = jax.tree_util.tree_map(lambda m: m / int(num_minibatches), metrics_total)
    return grads, loss, metrics


def train_batch_parameter_shift_binomial(
    flat_model: PyTree,
    flat_opt_state: PyTree,
    treedef_model: PyTree,
    treedef_opt_state: PyTree,
    batch: Batch,
    num_minibatches: Int,
    key_shot_noise: jax.random.PRNGKey,
    optimizer,
):
    old_model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    old_opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)
    old_params = old_model.get_params()

    grads, loss, metrics = accumulate_parameter_shift_gradients_scan(
        old_model,
        batch,
        key_shot_noise,
        num_minibatches,
    )

    updates, new_opt_state = optimizer.update(grads, old_opt_state, old_params)
    new_params = optax.apply_updates(old_params, updates)
    new_model = eqx.tree_at(where=lambda model: model.params, pytree=old_model, replace=new_params)

    flat_new_model = jax.tree_util.tree_leaves(new_model)
    flat_new_opt_state = jax.tree_util.tree_leaves(new_opt_state)

    return loss, metrics, flat_new_model, flat_new_opt_state


@eqx.filter_jit
def train_batch(
    flat_model: PyTree,
    flat_opt_state: PyTree,
    treedef_model: PyTree,
    treedef_opt_state: PyTree,
    batch: Batch,
    num_minibatches: Int,
    key_shot_noise: jax.random.PRNGKey,
    optimizer,
    loss_fn: Callable = smooth_hinge_loss_PsiG,
):
    old_model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    old_opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)
    old_params = old_model.get_params()

    grads, loss, metrics = accumulate_gradients_scan(
        old_model,
        batch,
        key_shot_noise,
        num_minibatches,
        loss_fn,
    )

    updates, new_opt_state = optimizer.update(grads.params, old_opt_state, old_params)
    new_params = optax.apply_updates(old_params, updates)
    new_model = eqx.tree_at(where=lambda model: model.params, pytree=old_model, replace=new_params)

    flat_new_model = jax.tree_util.tree_leaves(new_model)
    flat_new_opt_state = jax.tree_util.tree_leaves(new_opt_state)

    return loss, metrics, flat_new_model, flat_new_opt_state


@eqx.filter_jit
def evaluate_batch(model: ErrorMitigatedPQCModel, batch: Batch, rng_batch: jax.random.PRNGKey):
    rng_batch_noise = jax.random.split(rng_batch, num=batch.inputs.shape[0])
    preds = eqx.filter_vmap(model, in_axes=(0, 0))(batch.inputs, rng_batch_noise)
    metrics = get_metrics(preds, batch)
    return preds, metrics
