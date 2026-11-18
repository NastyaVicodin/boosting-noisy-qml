"""PEC overhead and effective-shot configuration helpers."""

from __future__ import annotations

from typing import Mapping


def resolve_n_shots_eff(conf: Mapping[str, object]) -> int:
    """Canonicalize effective-shot config.

    Preferred key: ``n_shots_eff``.  ``n_shots`` is retained as a legacy alias.
    """

    if "n_shots_eff" in conf:
        n_shots_eff = int(conf["n_shots_eff"])
    elif "n_shots" in conf:
        n_shots_eff = int(conf["n_shots"])
    else:
        raise KeyError("Expected `n_shots_eff` (preferred) or legacy `n_shots` in config.")

    if n_shots_eff <= 0:
        raise ValueError(f"n_shots_eff must be positive, got {n_shots_eff}.")
    return n_shots_eff


def _resolve_noise_strength_value(conf: Mapping[str, object], key: str, fallback: float) -> float:
    value = conf.get(key, fallback)
    if value is None:
        value = fallback
    value = float(value)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{key} must lie in [0, 1), got {value}.")
    return value


def resolve_pec_noise_strengths(conf: Mapping[str, object]) -> tuple[float, float]:
    """Resolve one- and two-qubit depolarizing strengths for PEC accounting."""

    scalar_fallback = _resolve_noise_strength_value(conf, "noise_strength", 0.0)
    noise_strength_1q = _resolve_noise_strength_value(conf, "noise_strength_1q", scalar_fallback)
    noise_strength_2q = _resolve_noise_strength_value(conf, "noise_strength_2q", scalar_fallback)
    return noise_strength_1q, noise_strength_2q


def pec_sampling_overhead_depolarizing(
    noise_strength_1q=None,
    noise_strength_2q=None,
    l1=None,
    l2=None,
    **kwargs,
) -> float:
    """Local-depolarizing PEC overhead ``gamma^2`` from the manuscript formulas."""

    if noise_strength_1q is None and "noise_strength" in kwargs:
        noise_strength_1q = kwargs.pop("noise_strength")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    if noise_strength_1q is None:
        raise TypeError("noise_strength_1q is required.")

    if l2 is None:
        if noise_strength_2q is None or l1 is None:
            raise TypeError(
                "Expected either (noise_strength_1q, noise_strength_2q, l1, l2) "
                "or legacy (noise_strength, l1, l2)."
            )
        legacy_l1 = noise_strength_2q
        legacy_l2 = l1
        noise_strength_2q = noise_strength_1q
        l1 = legacy_l1
        l2 = legacy_l2

    noise_strength_1q = float(noise_strength_1q)
    noise_strength_2q = float(noise_strength_2q)
    if not (0.0 <= noise_strength_1q < 1.0):
        raise ValueError(f"noise_strength_1q must lie in [0, 1), got {noise_strength_1q}.")
    if not (0.0 <= noise_strength_2q < 1.0):
        raise ValueError(f"noise_strength_2q must lie in [0, 1), got {noise_strength_2q}.")

    gamma1 = ((1.0 + noise_strength_1q / 2.0) / (1.0 - noise_strength_1q)) ** int(l1)
    gamma2 = ((1.0 + 7.0 * noise_strength_2q / 8.0) / (1.0 - noise_strength_2q)) ** int(l2)
    return float((gamma1 * gamma2) ** 2)


def resolve_pec_gamma_sq(conf: Mapping[str, object], n_qubits: int, n_layers: int) -> float:
    """Resolve PEC sampling overhead ``gamma^2`` for raw-shot accounting."""

    if "pec_gamma_sq" in conf:
        gamma_sq = float(conf["pec_gamma_sq"])
    else:
        noise_strength_1q, noise_strength_2q = resolve_pec_noise_strengths(conf)
        if noise_strength_1q <= 0.0 and noise_strength_2q <= 0.0:
            gamma_sq = 1.0
        else:
            ansatz_kind = str(conf.get("ansatz_kind", "legacy_reuploading"))
            # Ring-CZ connectivity: n_qubits pairs for n>2, n_qubits-1 for n==2.
            n_cz_pairs = int(n_qubits) if int(n_qubits) > 2 else max(1, int(n_qubits) - 1)
            if ansatz_kind == "paper_modular":
                n_reuploads = int(conf.get("n_reuploads", 1))
                if n_reuploads <= 0:
                    raise ValueError(f"n_reuploads must be positive, got {n_reuploads}.")
                total_blocks = int(n_reuploads * n_layers)
                l1_default = int(6 * total_blocks * n_qubits)
                l2_default = int(total_blocks * n_cz_pairs)
            else:
                l1_default = int((4 * n_layers + 3) * n_qubits)
                l2_default = int(n_layers * n_cz_pairs)
            l1 = int(conf.get("pec_l1", l1_default))
            l2 = int(conf.get("pec_l2", l2_default))
            gamma_sq = pec_sampling_overhead_depolarizing(
                noise_strength_1q=noise_strength_1q,
                noise_strength_2q=noise_strength_2q,
                l1=l1,
                l2=l2,
            )

    if gamma_sq < 1.0:
        raise ValueError(f"pec_gamma_sq must be >= 1.0, got {gamma_sq}.")
    return float(gamma_sq)
