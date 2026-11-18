"""Small NumPy compatibility helpers used across training scripts."""

from typing import Any

import numpy as np


def patch_numpy_complex_warning() -> bool:
    """Restore ``np.ComplexWarning`` for libraries that still read it.

    NumPy 2 moved ``ComplexWarning`` under ``np.exceptions``. TensorCircuit
    0.12 still accesses the old top-level name during import.
    """
    if hasattr(np, "ComplexWarning"):
        return False
    complex_warning = getattr(getattr(np, "exceptions", None), "ComplexWarning", None)
    if complex_warning is None:
        return False
    np.ComplexWarning = complex_warning  # type: ignore[attr-defined]
    return True


def patch_numpy_reshape_newshape_kw() -> bool:
    """Support legacy ``np.reshape(..., newshape=...)`` calls on NumPy>=2.

    Some TensorCircuit versions still pass the ``newshape`` keyword, which was
    removed in newer NumPy APIs. This helper adds a lightweight adapter.
    Returns ``True`` only when a patch was applied.
    """
    try:
        np.reshape(np.zeros((1,)), newshape=(1,))
        return False
    except TypeError:
        pass

    original_reshape = np.reshape

    def _reshape_compat(a: Any, *args: Any, **kwargs: Any):
        if "newshape" in kwargs and "shape" not in kwargs:
            kwargs["shape"] = kwargs.pop("newshape")
        return original_reshape(a, *args, **kwargs)

    np.reshape = _reshape_compat  # type: ignore[assignment]
    return True


def patch_numpy_for_tensorcircuit() -> dict[str, bool]:
    """Apply NumPy compatibility patches required before importing TensorCircuit."""
    return {
        "complex_warning": patch_numpy_complex_warning(),
        "reshape_newshape_kw": patch_numpy_reshape_newshape_kw(),
    }
