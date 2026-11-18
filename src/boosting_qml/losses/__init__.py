"""Loss and metric helpers."""

from .margins import classification_error, empirical_margin_loss, normalized_margin, raw_margin

try:
    from .surrogates import (
        PsiG,
        classification_error_from_scores,
        empirical_hinge_margin_loss,
        smooth_hinge_loss_PsiG,
    )
except Exception:  # pragma: no cover - JAX not installed in lightweight test env
    PsiG = None  # type: ignore[assignment]
    classification_error_from_scores = None  # type: ignore[assignment]
    empirical_hinge_margin_loss = None  # type: ignore[assignment]
    smooth_hinge_loss_PsiG = None  # type: ignore[assignment]

__all__ = [
    "PsiG",
    "classification_error",
    "classification_error_from_scores",
    "empirical_margin_loss",
    "empirical_hinge_margin_loss",
    "normalized_margin",
    "raw_margin",
    "smooth_hinge_loss_PsiG",
]
