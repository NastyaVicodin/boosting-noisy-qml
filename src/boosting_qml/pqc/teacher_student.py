import jax
from jax import config

import jax.numpy as jnp
DTYPE=jnp.float32

import chex
import numpy as np
from numpy_compat import patch_numpy_for_tensorcircuit

patch_numpy_for_tensorcircuit()

from scipy.stats import qmc

from functools import partial
from pathlib import Path
import tensorcircuit as tc
import equinox as eqx

import types
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Union, Sequence, List, NamedTuple, Optional, Tuple, Any, Literal, TypeVar


def reuploading_circuit(
    n_qubits: int,
    n_layers: int,
    params: jnp.ndarray,
    X: jnp.ndarray,
):
    """
    Reuploading PQC with qml_layer-style flattened params.

    Expected params shape: (4 * n_layers + 3, n_qubits).
    Layer layout:
      - params[4*l + 0, q]: input scale lambda_{l,q}
      - params[4*l + 1, q]: RX variational angle
      - params[4*l + 2, q]: RY variational angle
      - params[4*l + 3, q]: RZ variational angle
    Final block:
      - params[4*n_layers + 0..2, q]: final RX/RY/RZ.
    """
    expected_shape = (4 * n_layers + 3, n_qubits)
    if params.shape != expected_shape:
        raise ValueError(f"Expected params shape {expected_shape}, got {tuple(params.shape)}.")
    if X.shape[0] != n_qubits:
        raise ValueError(f"Expected X.shape[0] == {n_qubits}, got {X.shape[0]}.")

    circuit = tc.Circuit(n_qubits)

    for l in range(n_layers):
        # Variational part
        for q in range(n_qubits):
            circuit.rx(q, theta=params[4 * l + 1, q])
            circuit.ry(q, theta=params[4 * l + 2, q])
            circuit.rz(q, theta=params[4 * l + 3, q])

        # Entangling part (ring CZ)
        for q in range(n_qubits - 1):
            circuit.cz(q, q + 1)
        if n_qubits != 2:
            circuit.cz(n_qubits - 1, 0)

        # Encoding part
        for q in range(n_qubits):
            encoded = X[q] * params[4 * l + 0, q]
            circuit.rx(q, theta=encoded)

    # Final variational block
    for q in range(n_qubits):
        circuit.rx(q, theta=params[4 * n_layers + 0, q])
        circuit.ry(q, theta=params[4 * n_layers + 1, q])
        circuit.rz(q, theta=params[4 * n_layers + 2, q])

    return circuit

class NoiselessPQCModel(eqx.Module):
    params: jax.Array
    n_qubits: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)

    def __init__(
        self, 
        n_qubits: int, 
        n_layers: int, 
        key: jax.random.PRNGKey
    ):
        # self.key, *subkeys = jax.random.split(key=key, num=2)

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        key_params, _ = jax.random.split(key)

        self.params = jax.lax.stop_gradient(jax.random.normal(
            key=key_params, shape=(4 * n_layers + 3, n_qubits), dtype=DTYPE
        ))
    
    def get_params(self):
      return self.params

    def __call__(
      self, 
      inputs: jax.Array, 
    ):
      circuit = reuploading_circuit(self.n_qubits, self.n_layers, self.params, inputs)
      return jnp.real(circuit.expectation_ps(z=np.arange(self.n_qubits)))


def teacher_params_from_seed(
    n_qubits: int,
    n_layers: int,
    data_generation_seed: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Reconstruct the exact fixed teacher parameters from generation seed."""
    key = jax.random.PRNGKey(int(data_generation_seed))
    _, rng_params = jax.random.split(key)
    key_params, _ = jax.random.split(rng_params)
    params = jax.random.normal(
        key=key_params,
        shape=(4 * int(n_layers) + 3, int(n_qubits)),
        dtype=dtype,
    )
    return jax.lax.stop_gradient(params)


def teacher_class_counts(
    *,
    n_qubits: int,
    n_layers: int,
    n_grid_points: int,
    margin: float,
    data_generation_seed: int,
    batch_size: int = 512,
) -> dict[str, int]:
    """Count margin-filtered classes using production teacher conventions.

    This uses the same ``NoiselessPQCModel``, global-Z observable, SciPy Latin
    hypercube, PRNG splitting, and ``(expectation + 1) / 2`` convention as
    :class:`Teacher_PQC_lhs`. It avoids sampling or caching because screening
    must depend only on constructibility.
    """
    key = jax.random.PRNGKey(int(data_generation_seed))
    _, rng_params = jax.random.split(key)
    model = NoiselessPQCModel(int(n_qubits), int(n_layers), rng_params)
    sampler = qmc.LatinHypercube(d=int(n_qubits), seed=int(data_generation_seed))
    sample = sampler.random(n=int(n_grid_points))
    points = qmc.scale(
        sample,
        [0.0] * int(n_qubits),
        [2.0 * np.pi] * int(n_qubits),
    )

    @jax.jit
    def evaluate_batch(current_model, current_points):
      return jax.vmap(current_model)(current_points)

    lower = 0.5 - float(margin) / 2.0
    upper = 0.5 + float(margin) / 2.0
    negative_count = 0
    positive_count = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, int(n_grid_points), batch_size):
      stop = min(start + batch_size, int(n_grid_points))
      expectations = evaluate_batch(model, jnp.asarray(points[start:stop], dtype=DTYPE))
      probabilities = np.asarray((expectations + 1.0) / 2.0)
      negative_count += int(np.sum(probabilities < lower))
      positive_count += int(np.sum(probabilities >= upper))
    return {
        "negative_candidate_count": negative_count,
        "positive_candidate_count": positive_count,
        "rejected_margin_count": int(n_grid_points) - negative_count - positive_count,
    }

class Teacher_PQC_lhs:

  def __init__(self, n_qubits, n_layers, n_grid_points, margin, n_data_points, data_generation_seed):
    self.n_qubits = n_qubits
    self.n_layers = n_layers
    self.n_grid_points = n_grid_points
    self.margin = margin
    self.n_data_points = n_data_points
    self.data_generation_seed = data_generation_seed
    self.grid_points, self.grid_values = self.generate_grid()
    self.labels = self.generate_labels(self.grid_values, self.margin)

    self.samples = self.generate_samples(self.labels, self.n_data_points)
    self.X = self.samples["X"]
    self.y = self.samples["y"]

  def generate_grid(self):
    key = jax.random.PRNGKey(self.data_generation_seed)
    rng, rng_params = jax.random.split(key)

    noiseless_model = NoiselessPQCModel(self.n_qubits, self.n_layers, rng_params) 
    sampler = qmc.LatinHypercube(d=self.n_qubits, seed=self.data_generation_seed)
    sample = sampler.random(n=self.n_grid_points)

    l_bounds = [0] * self.n_qubits
    u_bounds = [2 * np.pi] * self.n_qubits
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    fct = np.zeros(self.n_grid_points)

    @jax.jit
    def evaluate_point(model, point):
      return (model(point) + 1) / 2

    for idx, samp in enumerate(sample_scaled):
      fct[idx] = evaluate_point(noiseless_model, jnp.array(samp))

    return sample_scaled, fct

  def generate_labels(self, values, margin):
    labels = np.zeros_like(values)

    for idx, y in enumerate(values):
      if y < 0.5 - margin / 2:
        labels[idx] = -1
      elif y >= 0.5 + margin / 2:
        labels[idx] = 1

    return labels

  def generate_samples(self, labels, n_data_points, no_doubles=False):
    """
    Sample a balanced binary dataset from teacher-labeled grid points.

    Returns a dictionary with:
      - class_indices: [negative_class_indices, positive_class_indices]
      - indices: shuffled global indices used in the final dataset
      - X: features of shape (2*n_data_points, n_qubits)
      - y: labels in {-1, +1} of shape (2*n_data_points,)
    """
    labels = np.asarray(labels)
    neg_pool = np.where(labels < 0)[0]
    pos_pool = np.where(labels > 0)[0]

    if neg_pool.size == 0 or pos_pool.size == 0:
      raise ValueError(
          "Could not form a binary dataset: one class is empty after margin thresholding. "
          "Try reducing margin or increasing n_grid_points."
      )

    if no_doubles and (neg_pool.size < n_data_points or pos_pool.size < n_data_points):
      raise ValueError(
          "Not enough unique points per class for no_doubles=True. "
          "Increase n_grid_points or lower n_data_points."
      )

    rng = np.random.default_rng(self.data_generation_seed)
    replace = not no_doubles
    neg_indices = rng.choice(neg_pool, size=n_data_points, replace=replace)
    pos_indices = rng.choice(pos_pool, size=n_data_points, replace=replace)

    all_indices = np.concatenate([neg_indices, pos_indices], axis=0)
    perm = rng.permutation(all_indices.shape[0])
    all_indices = all_indices[perm]

    x_samples = np.asarray(self.grid_points[all_indices], dtype=np.float32)
    y_samples = np.asarray(labels[all_indices], dtype=np.int8)
    y_samples = np.where(y_samples > 0, 1, -1).astype(np.int8)

    return {
        "class_indices": [neg_indices.tolist(), pos_indices.tolist()],
        "indices": all_indices,
        "X": x_samples,
        "y": y_samples,
    }

  def get_train_test_split(self, test_fraction=0.2):
    """Return a deterministic train/test split from generated samples."""
    if not (0.0 < test_fraction < 1.0):
      raise ValueError("test_fraction must be in (0, 1).")

    n_samples = self.X.shape[0]
    n_test = int(round(n_samples * test_fraction))
    n_test = max(1, min(n_samples - 1, n_test))

    rng = np.random.default_rng(self.data_generation_seed + 1)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return (
        self.X[train_idx],
        self.y[train_idx],
        self.X[test_idx],
        self.y[test_idx],
    )


def _float_token(value: float, precision: int = 8) -> str:
  """Return a filename-safe deterministic token for floating-point values."""
  token = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
  if token == "" or token == "-0":
    token = "0"
  return token.replace("-", "m").replace(".", "p")


def teacher_student_dataset_filename(
    n_qubits: int,
    teacher_n_layers: int,
    train_size: int,
    test_size: int,
    teacher_margin: float,
    teacher_n_grid_points: int,
    data_generation_seed: int,
    no_doubles: bool = True,
    cache_version: int = 2,
) -> str:
  """Deterministic dataset filename that encodes all generation knobs."""
  margin_token = _float_token(float(teacher_margin), precision=8)
  unique_token = "unique1" if bool(no_doubles) else "unique0"
  return (
      f"teacher_student_v{int(cache_version)}_"
      f"nq{int(n_qubits)}_"
      f"tl{int(teacher_n_layers)}_"
      f"grid{int(teacher_n_grid_points)}_"
      f"m{margin_token}_"
      f"ntr{int(train_size)}_"
      f"nte{int(test_size)}_"
      f"{unique_token}_"
      f"seed{int(data_generation_seed)}.npz"
  )


def load_or_generate_teacher_student_binary_dataset(
    n_qubits: int,
    teacher_n_layers: int,
    train_size: int,
    test_size: int,
    teacher_margin: float,
    teacher_n_grid_points: int,
    data_generation_seed: int,
    save_dir: str = "datasets/teacher_student",
    use_cache: bool = True,
    overwrite_cache: bool = False,
    no_doubles: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load teacher-student data from cache or generate and save it.

  The filename fully captures all dataset-defining parameters so experiments are
  reproducible and datasets can be regenerated from the filename alone.
  """
  if int(train_size) < 1 or int(test_size) < 1:
    raise ValueError("train_size and test_size must both be positive.")
  if int(teacher_n_grid_points) < 2:
    raise ValueError("teacher_n_grid_points must be at least 2.")
  if not (0.0 <= float(teacher_margin) < 1.0):
    raise ValueError("teacher_margin must be in [0, 1).")

  total_samples = int(train_size + test_size)
  if total_samples < 2:
    raise ValueError("Total samples must be at least 2.")

  n_data_points_per_class = int(np.ceil(total_samples / 2.0))
  test_fraction = float(test_size) / float(total_samples)

  save_path = Path(save_dir)
  save_path.mkdir(parents=True, exist_ok=True)

  file_name = teacher_student_dataset_filename(
      n_qubits=n_qubits,
      teacher_n_layers=teacher_n_layers,
      train_size=train_size,
      test_size=test_size,
      teacher_margin=teacher_margin,
      teacher_n_grid_points=teacher_n_grid_points,
      data_generation_seed=data_generation_seed,
      no_doubles=no_doubles,
  )
  file_path = save_path / file_name

  should_load = bool(use_cache) and file_path.exists() and not bool(overwrite_cache)
  if should_load:
    payload = np.load(file_path)
    X_train_np = np.asarray(payload["X_train"], dtype=np.float32)
    y_train_np = np.asarray(payload["y_train"], dtype=np.float32)
    X_test_np = np.asarray(payload["X_test"], dtype=np.float32)
    y_test_np = np.asarray(payload["y_test"], dtype=np.float32)
  else:
    teacher_dataset = Teacher_PQC_lhs(
        n_qubits=int(n_qubits),
        n_layers=int(teacher_n_layers),
        n_grid_points=int(teacher_n_grid_points),
        margin=float(teacher_margin),
        n_data_points=int(n_data_points_per_class),
        data_generation_seed=int(data_generation_seed),
    )

    # By default, teacher-student data must use unique teacher grid points.
    # In high dimensions the margin-filtered accepted pool can become tiny;
    # sampling with replacement then creates fake-easy datasets with many
    # duplicates. no_doubles=True turns that silent failure into a clear error.
    teacher_dataset.samples = teacher_dataset.generate_samples(
        teacher_dataset.labels,
        teacher_dataset.n_data_points,
        no_doubles=bool(no_doubles),
    )
    teacher_dataset.X = teacher_dataset.samples["X"]
    teacher_dataset.y = teacher_dataset.samples["y"]

    X_train_np, y_train_np, X_test_np, y_test_np = teacher_dataset.get_train_test_split(
        test_fraction=test_fraction
    )

    if X_train_np.shape[0] < int(train_size) or X_test_np.shape[0] < int(test_size):
      raise ValueError(
          "Teacher-student split did not produce enough samples for the requested "
          f"train_size={train_size}, test_size={test_size}. "
          "Increase total samples or adjust split settings."
      )

    X_train_np = np.asarray(X_train_np[:train_size], dtype=np.float32)
    y_train_np = np.asarray(y_train_np[:train_size], dtype=np.float32)
    X_test_np = np.asarray(X_test_np[:test_size], dtype=np.float32)
    y_test_np = np.asarray(y_test_np[:test_size], dtype=np.float32)

    teacher_params_np = np.asarray(
        teacher_params_from_seed(
            n_qubits=n_qubits,
            n_layers=teacher_n_layers,
            data_generation_seed=data_generation_seed,
            dtype=DTYPE,
        ),
        dtype=np.float32,
    )

    if bool(use_cache) or bool(overwrite_cache):
      np.savez(
          file_path,
          X_train=X_train_np,
          y_train=y_train_np,
          X_test=X_test_np,
          y_test=y_test_np,
          teacher_params=teacher_params_np,
          n_qubits=np.int32(n_qubits),
          teacher_n_layers=np.int32(teacher_n_layers),
          train_size=np.int32(train_size),
          test_size=np.int32(test_size),
          teacher_margin=np.float64(teacher_margin),
          teacher_n_grid_points=np.int32(teacher_n_grid_points),
          data_generation_seed=np.int32(data_generation_seed),
          n_data_points_per_class=np.int32(n_data_points_per_class),
          no_doubles=np.bool_(no_doubles),
          train_unique_count=np.int32(np.unique(X_train_np, axis=0).shape[0]),
          test_unique_count=np.int32(np.unique(X_test_np, axis=0).shape[0]),
          combined_unique_count=np.int32(np.unique(np.concatenate([X_train_np, X_test_np], axis=0), axis=0).shape[0]),
          train_duplicate_fraction=np.float64(1.0 - np.unique(X_train_np, axis=0).shape[0] / max(1, X_train_np.shape[0])),
          test_duplicate_fraction=np.float64(1.0 - np.unique(X_test_np, axis=0).shape[0] / max(1, X_test_np.shape[0])),
          accepted_negative_pool_size=np.int32(np.sum(teacher_dataset.labels < 0)),
          accepted_positive_pool_size=np.int32(np.sum(teacher_dataset.labels > 0)),
          test_fraction=np.float64(test_fraction),
          cache_version=np.int32(2),
      )

  return (
      jnp.asarray(X_train_np, dtype=dtype),
      jnp.asarray(y_train_np, dtype=dtype),
      jnp.asarray(X_test_np, dtype=dtype),
      jnp.asarray(y_test_np, dtype=dtype),
  )

__all__ = [
    "DTYPE",
    "reuploading_circuit",
    "NoiselessPQCModel",
    "teacher_params_from_seed",
    "Teacher_PQC_lhs",
    "teacher_student_dataset_filename",
    "load_or_generate_teacher_student_binary_dataset",
]
