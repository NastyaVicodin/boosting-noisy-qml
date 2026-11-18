"""Deterministic toy datasets and FMNIST preprocessing helpers."""

from __future__ import annotations

import gzip
from pathlib import Path
import struct
from typing import Iterable, Literal, Tuple

import numpy as np

from boosting_qml.types import BinaryDataset, BinaryDatasetSplit


def make_toy_binary_classification(
    n_samples: int = 320,
    n_features: int = 5,
    seed: int = 7,
    label_noise: float = 0.0,
) -> BinaryDataset:
    """Create a deterministic linear toy problem with optional label flips."""

    if n_samples < 4:
        raise ValueError("n_samples must be at least 4.")
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    if not (0.0 <= label_noise < 0.5):
        raise ValueError("label_noise must lie in [0, 0.5).")

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    w_true = rng.normal(size=(n_features,))
    raw_score = x @ w_true + 0.25 * rng.normal(size=(n_samples,))
    y = np.where(raw_score >= 0.0, 1, -1).astype(np.int8)

    if label_noise > 0.0:
        flips = rng.random(size=n_samples) < label_noise
        y[flips] *= -1

    return BinaryDataset(X=x, y=y)


def train_test_split_binary(
    dataset: BinaryDataset,
    test_fraction: float = 0.3,
    seed: int = 7,
) -> Tuple[BinaryDataset, BinaryDataset]:
    """Deterministic train/test split for binary datasets."""

    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must lie in (0, 1).")

    n_samples = dataset.n_samples
    n_test = int(round(test_fraction * n_samples))
    n_test = max(1, min(n_samples - 1, n_test))

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_data = BinaryDataset(X=dataset.X[train_idx], y=dataset.y[train_idx])
    test_data = BinaryDataset(X=dataset.X[test_idx], y=dataset.y[test_idx])
    return train_data, test_data



def _allocate_binary_validation_counts(
    y: np.ndarray,
    n_validation: int,
) -> tuple[int, int] | None:
    y = np.asarray(y, dtype=np.int8)
    n_total = int(y.shape[0])
    neg_count = int(np.sum(y < 0))
    pos_count = int(np.sum(y > 0))

    if neg_count < 2 or pos_count < 2:
        return None
    if n_validation < 2 or n_validation > n_total - 2:
        return None

    n_validation_pos = int(round(n_validation * pos_count / n_total))
    lower = max(1, n_validation - (neg_count - 1))
    upper = min(pos_count - 1, n_validation - 1)
    if lower > upper:
        return None

    n_validation_pos = int(np.clip(n_validation_pos, lower, upper))
    n_validation_neg = int(n_validation - n_validation_pos)
    if not (1 <= n_validation_neg <= neg_count - 1):
        return None
    return n_validation_neg, n_validation_pos


def train_validation_split_binary(
    dataset: BinaryDataset,
    validation_fraction: float = 0.0,
    seed: int = 11,
    stratify: bool = True,
) -> tuple[BinaryDataset, BinaryDataset | None]:
    """Deterministic train/validation split for already-constructed train data."""

    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError('validation_fraction must lie in [0, 1).')
    if validation_fraction <= 0.0:
        return dataset, None

    n_samples = dataset.n_samples
    if n_samples < 2:
        raise ValueError('Need at least two samples to create a validation split.')

    n_validation = int(round(validation_fraction * n_samples))
    n_validation = max(1, min(n_samples - 1, n_validation))

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    train_idx: np.ndarray
    validation_idx: np.ndarray

    counts = _allocate_binary_validation_counts(dataset.y, n_validation) if stratify else None
    if counts is None:
        rng.shuffle(indices)
        validation_idx = indices[:n_validation]
        train_idx = indices[n_validation:]
    else:
        n_validation_neg, n_validation_pos = counts
        neg_idx = np.where(dataset.y < 0)[0]
        pos_idx = np.where(dataset.y > 0)[0]
        rng.shuffle(neg_idx)
        rng.shuffle(pos_idx)
        validation_idx = np.concatenate(
            [neg_idx[:n_validation_neg], pos_idx[:n_validation_pos]],
            axis=0,
        )
        train_idx = np.concatenate(
            [neg_idx[n_validation_neg:], pos_idx[n_validation_pos:]],
            axis=0,
        )
        rng.shuffle(validation_idx)
        rng.shuffle(train_idx)

    train_data = BinaryDataset(X=dataset.X[train_idx], y=dataset.y[train_idx])
    validation_data = BinaryDataset(X=dataset.X[validation_idx], y=dataset.y[validation_idx])
    return train_data, validation_data


def _default_fmnist_cache_search_dirs() -> Tuple[Path, ...]:
    """Default cache roots, including legacy boosting_jax artifacts."""

    project_root = Path(__file__).resolve().parents[3]  # .../boosting_new
    return (
        project_root / "datasets" / "fmnist",
        project_root.parent / "boosting_jax" / "datasets" / "fmnist",
    )


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using training statistics only."""

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    train_x_std = (train_x - mean) / std
    test_x_std = (test_x - mean) / std
    return train_x_std, test_x_std, mean, std


def _fit_pca_via_svd(train_x: np.ndarray, n_features: int) -> np.ndarray:
    """Fit PCA components with SVD (rows=samples, cols=features)."""

    if n_features < 1:
        raise ValueError("n_features must be >= 1.")
    if n_features > train_x.shape[1]:
        raise ValueError("n_features cannot exceed input dimensionality.")

    _, _, vh = np.linalg.svd(train_x, full_matrices=False)
    return vh[:n_features]


def _transform_with_components(x: np.ndarray, components: np.ndarray) -> np.ndarray:
    return x @ components.T


def _load_fashion_mnist_raw() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST from keras/tensorflow if available.

    Raises:
        RuntimeError: If neither provider is available.
    """

    local = _load_fashion_mnist_raw_from_local_idx_cache()
    if local is not None:
        return local

    try:
        from keras.datasets import fashion_mnist  # type: ignore

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import fashion_mnist  # type: ignore

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return x_train, y_train, x_test, y_test
    except Exception as exc:
        raise RuntimeError(
            "Could not load raw Fashion-MNIST. "
            "Install `keras` or `tensorflow`, or provide a compatible cached "
            "binary_fmnist_*.npz file in one of the search directories."
        ) from exc


def _fashion_mnist_local_cache_dirs() -> Tuple[Path, ...]:
    """Candidate local directories containing gzip-compressed IDX files."""

    home_dir = Path.home()
    return (
        home_dir / ".keras" / "datasets" / "fashion-mnist",
    )


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        content = f.read()
    if len(content) < 8:
        raise RuntimeError(f"Invalid IDX label file: {path}")
    magic, n_items = struct.unpack(">II", content[:8])
    if magic != 2049:
        raise RuntimeError(f"Unexpected label magic number {magic} in {path}")
    labels = np.frombuffer(content, dtype=np.uint8, offset=8)
    if labels.shape[0] != n_items:
        raise RuntimeError(f"Label count mismatch in {path}: header={n_items}, actual={labels.shape[0]}")
    return labels.copy()


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        content = f.read()
    if len(content) < 16:
        raise RuntimeError(f"Invalid IDX image file: {path}")
    magic, n_items, n_rows, n_cols = struct.unpack(">IIII", content[:16])
    if magic != 2051:
        raise RuntimeError(f"Unexpected image magic number {magic} in {path}")
    flat = np.frombuffer(content, dtype=np.uint8, offset=16)
    expected = n_items * n_rows * n_cols
    if flat.shape[0] != expected:
        raise RuntimeError(f"Image count mismatch in {path}: expected={expected}, actual={flat.shape[0]}")
    return flat.reshape(n_items, n_rows, n_cols).copy()


def _load_fashion_mnist_raw_from_local_idx_cache() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load Fashion-MNIST directly from local IDX gzip files when available."""

    for cache_dir in _fashion_mnist_local_cache_dirs():
        train_images = cache_dir / "train-images-idx3-ubyte.gz"
        train_labels = cache_dir / "train-labels-idx1-ubyte.gz"
        test_images = cache_dir / "t10k-images-idx3-ubyte.gz"
        test_labels = cache_dir / "t10k-labels-idx1-ubyte.gz"
        if not (train_images.exists() and train_labels.exists() and test_images.exists() and test_labels.exists()):
            continue
        try:
            x_train = _read_idx_images_gz(train_images)
            y_train = _read_idx_labels_gz(train_labels)
            x_test = _read_idx_images_gz(test_images)
            y_test = _read_idx_labels_gz(test_labels)
            return x_train, y_train, x_test, y_test
        except Exception:
            continue
    return None


def prepare_binary_fmnist(
    digit_a: int = 0,
    digit_b: int = 1,
    n_features: int = 6,
    n_samples_train: int = 4096,
    n_samples_test: int = 1024,
    random_seed: int = 42,
    save_dir: str = "datasets/fmnist",
    force_rebuild: bool = False,
    search_dirs: Iterable[str] = (),
) -> Tuple[BinaryDataset, BinaryDataset]:
    """
    Prepare a binary FMNIST split compatible with adaboost_jax.py preprocessing.

    Pipeline parity with the rough script:
    1) select two FMNIST classes and map to {-1,+1},
    2) flatten 28x28 images,
    3) standardize (fit on train, apply to test),
    4) PCA projection to n_features,
    5) deterministic sub-sampling without replacement.
    """

    if digit_a == digit_b:
        raise ValueError("digit_a and digit_b must be different classes.")
    if n_samples_train < 1 or n_samples_test < 1:
        raise ValueError("Sample counts must be positive.")

    file_name = (
        f"binary_fmnist_{digit_a}_{digit_b}_{n_features}_"
        f"{n_samples_train}_{n_samples_test}_{random_seed}.npz"
    )
    requested_save_dir = Path(save_dir)
    candidate_paths = [requested_save_dir / file_name]

    for root in _default_fmnist_cache_search_dirs():
        candidate_paths.append(root / file_name)
    for extra in search_dirs:
        candidate_paths.append(Path(extra) / file_name)

    cached_path = None
    if not force_rebuild:
        for path in candidate_paths:
            if path.exists():
                cached_path = path
                break

    if cached_path is not None:
        loaded = np.load(cached_path, allow_pickle=True)
        x_train = np.asarray(loaded["X_train"], dtype=np.float64)
        y_train = np.asarray(loaded["y_train"], dtype=np.int8)
        x_test = np.asarray(loaded["X_test"], dtype=np.float64)
        y_test = np.asarray(loaded["y_test"], dtype=np.int8)

        y_train = np.where(y_train > 0, 1, -1).astype(np.int8)
        y_test = np.where(y_test > 0, 1, -1).astype(np.int8)
        return BinaryDataset(x_train, y_train), BinaryDataset(x_test, y_test)

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_fashion_mnist_raw()

    train_mask = np.logical_or(y_train_raw == digit_a, y_train_raw == digit_b)
    test_mask = np.logical_or(y_test_raw == digit_a, y_test_raw == digit_b)
    x_train_raw = x_train_raw[train_mask]
    y_train_raw = y_train_raw[train_mask]
    x_test_raw = x_test_raw[test_mask]
    y_test_raw = y_test_raw[test_mask]

    y_train = np.where(y_train_raw == digit_a, -1, 1).astype(np.int8)
    y_test = np.where(y_test_raw == digit_a, -1, 1).astype(np.int8)

    train_flat = x_train_raw.reshape((-1, 28 * 28)).astype(np.float64)
    test_flat = x_test_raw.reshape((-1, 28 * 28)).astype(np.float64)
    train_std, test_std, mean, std = _standardize(train_flat, test_flat)
    components = _fit_pca_via_svd(train_std, n_features=n_features)

    x_train = _transform_with_components(train_std, components)
    x_test = _transform_with_components(test_std, components)

    if n_samples_train > len(x_train):
        raise ValueError(
            f"Requested n_samples_train={n_samples_train}, but only {len(x_train)} available "
            f"for classes ({digit_a}, {digit_b})."
        )
    if n_samples_test > len(x_test):
        raise ValueError(
            f"Requested n_samples_test={n_samples_test}, but only {len(x_test)} available "
            f"for classes ({digit_a}, {digit_b})."
        )

    rng = np.random.default_rng(random_seed)
    train_idx = rng.choice(np.arange(len(x_train)), size=n_samples_train, replace=False)
    test_idx = rng.choice(np.arange(len(x_test)), size=n_samples_test, replace=False)

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    requested_save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        requested_save_dir / file_name,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        standardize_mean=mean,
        standardize_std=std,
        pca_components=components,
    )

    return BinaryDataset(x_train, y_train), BinaryDataset(x_test, y_test)


def prepare_teacher_student_binary_dataset(
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
) -> Tuple[BinaryDataset, BinaryDataset]:
    """Prepare/reload teacher-student binary data with deterministic caching."""

    from boosting_qml.pqc.teacher_student import load_or_generate_teacher_student_binary_dataset

    x_train, y_train, x_test, y_test = load_or_generate_teacher_student_binary_dataset(
        n_qubits=int(n_qubits),
        teacher_n_layers=int(teacher_n_layers),
        train_size=int(train_size),
        test_size=int(test_size),
        teacher_margin=float(teacher_margin),
        teacher_n_grid_points=int(teacher_n_grid_points),
        data_generation_seed=int(data_generation_seed),
        save_dir=save_dir,
        use_cache=bool(use_cache),
        overwrite_cache=bool(overwrite_cache),
        no_doubles=bool(no_doubles),
    )

    train_data = BinaryDataset(
        X=np.asarray(x_train, dtype=np.float64),
        y=np.where(np.asarray(y_train) > 0.0, 1, -1).astype(np.int8),
    )
    test_data = BinaryDataset(
        X=np.asarray(x_test, dtype=np.float64),
        y=np.where(np.asarray(y_test) > 0.0, 1, -1).astype(np.int8),
    )
    return train_data, test_data


def prepare_class_conditional_gaussian_binary_dataset(
    n_features: int,
    n_samples_train: int,
    n_samples_test: int,
    random_seed: int = 42,
    variance: float = 0.8,
    save_dir: str = "datasets/gaussian_class_conditional",
    use_cache: bool = True,
    overwrite_cache: bool = False,
) -> Tuple[BinaryDataset, BinaryDataset]:
    """Prepare class-conditional Gaussian binary data used in phase-0.5."""

    if float(variance) <= 0.0:
        raise ValueError(f"variance must be > 0, got {variance}.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    variance_tag = f"{float(variance):.6f}".replace(".", "p")
    file_name = (
        "gaussian_class_conditional_"
        f"d{int(n_features)}_ntr{int(n_samples_train)}_nte{int(n_samples_test)}_"
        f"var{variance_tag}_seed{int(random_seed)}.npz"
    )
    file_path = save_path / file_name

    if use_cache and file_path.exists() and not overwrite_cache:
        data = np.load(file_path)
        train_data = BinaryDataset(
            X=np.asarray(data["X_train"], dtype=np.float64),
            y=np.where(np.asarray(data["y_train"]) > 0.0, 1, -1).astype(np.int8),
        )
        test_data = BinaryDataset(
            X=np.asarray(data["X_test"], dtype=np.float64),
            y=np.where(np.asarray(data["y_test"]) > 0.0, 1, -1).astype(np.int8),
        )
        return train_data, test_data

    rng = np.random.default_rng(int(random_seed))
    dims = np.arange(int(n_features), dtype=np.int32)
    mu_plus = np.mod((2.0 * np.pi / 16.0) * (dims % 8), 2.0 * np.pi)
    mu_minus = np.mod((2.0 * np.pi / 16.0) * (8 + (dims % 8)), 2.0 * np.pi)
    std = float(np.sqrt(float(variance)))

    def _sample_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=int(n_samples), replace=True)
        means = np.where(labels[:, None] > 0.0, mu_plus[None, :], mu_minus[None, :])
        features = rng.normal(loc=means, scale=std, size=(int(n_samples), int(n_features))).astype(np.float32)
        return features, labels.astype(np.float32)

    x_train_np, y_train_np = _sample_split(int(n_samples_train))
    x_test_np, y_test_np = _sample_split(int(n_samples_test))

    if use_cache:
        np.savez(
            file_path,
            X_train=x_train_np,
            y_train=y_train_np,
            X_test=x_test_np,
            y_test=y_test_np,
            mu_plus=mu_plus.astype(np.float32),
            mu_minus=mu_minus.astype(np.float32),
            variance=np.array([float(variance)], dtype=np.float32),
            random_seed=np.array([int(random_seed)], dtype=np.int32),
        )

    train_data = BinaryDataset(
        X=np.asarray(x_train_np, dtype=np.float64),
        y=np.where(np.asarray(y_train_np) > 0.0, 1, -1).astype(np.int8),
    )
    test_data = BinaryDataset(
        X=np.asarray(x_test_np, dtype=np.float64),
        y=np.where(np.asarray(y_test_np) > 0.0, 1, -1).astype(np.int8),
    )
    return train_data, test_data


def prepare_phase05_binary_data(
    conf: dict,
    dataset_kind: Literal["teacher_student", "fmnist", "gaussian_class_conditional"] | None = None,
) -> Tuple[BinaryDataset, BinaryDataset]:
    """Unified data entrypoint for phase-0.5 experiments."""

    kind = conf.get("dataset_kind", "teacher_student") if dataset_kind is None else dataset_kind

    if kind == "teacher_student":
        return prepare_teacher_student_binary_dataset(
            n_qubits=int(conf["n_qubits"]),
            teacher_n_layers=int(conf.get("teacher_n_layers", conf["n_layers"])),
            train_size=int(conf["train_size"]),
            test_size=int(conf["test_size"]),
            teacher_margin=float(conf.get("teacher_margin", 0.2)),
            teacher_n_grid_points=int(conf.get("teacher_n_grid_points", 10000)),
            data_generation_seed=int(conf["data_generation_np_random_seed"]),
            save_dir=str(conf.get("teacher_student_save_dir", "datasets/teacher_student")),
            use_cache=bool(conf.get("teacher_student_use_cache", True)),
            overwrite_cache=bool(conf.get("teacher_student_overwrite_cache", False)),
            no_doubles=bool(conf.get("teacher_student_no_doubles", True)),
        )

    if kind == "fmnist":
        return prepare_binary_fmnist(
            digit_a=int(conf.get("digitA", 0)),
            digit_b=int(conf.get("digitB", 1)),
            n_features=int(conf["n_qubits"]),
            n_samples_train=int(conf["train_size"]),
            n_samples_test=int(conf["test_size"]),
            random_seed=int(conf["data_generation_np_random_seed"]),
            save_dir=str(conf.get("fmnist_save_dir", "datasets/fmnist")),
        )

    if kind == "gaussian_class_conditional":
        gaussian_n_features_raw = conf.get("gaussian_n_features", conf["n_qubits"])
        if gaussian_n_features_raw is None:
            gaussian_n_features_raw = conf["n_qubits"]
        gaussian_n_features = int(gaussian_n_features_raw)
        if gaussian_n_features <= 0:
            raise ValueError(f"gaussian_n_features must be positive, got {gaussian_n_features}.")

        ansatz_kind = str(conf.get("ansatz_kind", "legacy_reuploading"))
        if ansatz_kind == "legacy_reuploading" and gaussian_n_features != int(conf["n_qubits"]):
            raise ValueError(
                "For ansatz_kind='legacy_reuploading', gaussian_n_features must equal n_qubits "
                f"(got gaussian_n_features={gaussian_n_features}, n_qubits={int(conf['n_qubits'])})."
            )

        if ansatz_kind == "paper_modular" and str(conf.get("paper_data_chunk_mode", "cyclic")) == "strict":
            strict_required = int(3 * int(conf["n_qubits"]) * int(conf["n_layers"]))
            if gaussian_n_features != strict_required:
                raise ValueError(
                    "For ansatz_kind='paper_modular' with paper_data_chunk_mode='strict', "
                    "gaussian_n_features must satisfy D = 3 * n_qubits * n_layers "
                    f"(expected {strict_required}, got {gaussian_n_features})."
                )

        return prepare_class_conditional_gaussian_binary_dataset(
            n_features=gaussian_n_features,
            n_samples_train=int(conf["train_size"]),
            n_samples_test=int(conf["test_size"]),
            random_seed=int(conf["data_generation_np_random_seed"]),
            variance=float(conf.get("gaussian_variance", 0.8)),
            save_dir=str(conf.get("gaussian_save_dir", "datasets/gaussian_class_conditional")),
            use_cache=bool(conf.get("gaussian_use_cache", True)),
            overwrite_cache=bool(conf.get("gaussian_overwrite_cache", False)),
        )

    raise ValueError(
        f"Unsupported dataset_kind='{kind}'. Expected "
        "'teacher_student', 'fmnist', or 'gaussian_class_conditional'."
    )



def prepare_phase05_binary_data_split(
    conf: dict,
    dataset_kind: Literal["teacher_student", "fmnist", "gaussian_class_conditional"] | None = None,
) -> BinaryDatasetSplit:
    """Unified data entrypoint returning train/validation/test splits."""

    train_data, test_data = prepare_phase05_binary_data(conf, dataset_kind=dataset_kind)
    validation_fraction = float(conf.get('validation_fraction', 0.0))
    validation_seed = int(
        conf.get(
            'validation_seed',
            conf.get('data_generation_np_random_seed', conf.get('seed', 0)),
        )
    )
    validation_stratify = bool(conf.get('validation_stratify', True))
    train_split, validation_split = train_validation_split_binary(
        train_data,
        validation_fraction=validation_fraction,
        seed=validation_seed,
        stratify=validation_stratify,
    )
    return BinaryDatasetSplit(train=train_split, validation=validation_split, test=test_data)

def prepare_phase05_binary_data_arrays(
    conf: dict,
    dataset_kind: Literal["teacher_student", "fmnist", "gaussian_class_conditional"] | None = None,
    dtype=None,
):
    """Unified phase-0.5 data entrypoint returning JAX arrays."""

    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float32

    train_data, test_data = prepare_phase05_binary_data(conf, dataset_kind=dataset_kind)
    x_train = jnp.asarray(train_data.X, dtype=dtype)
    y_train = jnp.asarray(train_data.y, dtype=dtype)
    x_test = jnp.asarray(test_data.X, dtype=dtype)
    y_test = jnp.asarray(test_data.y, dtype=dtype)
    return x_train, y_train, x_test, y_test


def split_binary_train_validation_arrays(
    X,
    y,
    validation_fraction: float = 0.0,
    seed: int = 11,
    stratify: bool = True,
    dtype=None,
):
    """Deterministically split array data into fit-train and validation arrays."""

    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float32

    dataset = BinaryDataset(
        X=np.asarray(X, dtype=np.float64),
        y=np.where(np.asarray(y) > 0.0, 1, -1).astype(np.int8),
    )
    train_data, validation_data = train_validation_split_binary(
        dataset,
        validation_fraction=validation_fraction,
        seed=seed,
        stratify=stratify,
    )
    x_train = jnp.asarray(train_data.X, dtype=dtype)
    y_train = jnp.asarray(train_data.y, dtype=dtype)
    if validation_data is None:
        return x_train, y_train, None, None
    x_validation = jnp.asarray(validation_data.X, dtype=dtype)
    y_validation = jnp.asarray(validation_data.y, dtype=dtype)
    return x_train, y_train, x_validation, y_validation

