from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

CANONICAL_INFO_GROWTH_ROW_KEY = "info_growth_hardness"
CANONICAL_INFO_GROWTH_TRACE_KEY = "info_growth_hardness_by_step"
EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_ROW_KEY = "info_growth_hardness_no_rstar"
EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY = "info_growth_hardness_no_rstar_by_step"

BASE_MODEL_SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("disagreement", "disagreement_by_step"),
    ("residual_norm", "residual_norm_by_step"),
    ("hybrid_signal", "hybrid_signal_by_step"),
    ("u_disagreement", "u_disagreement_by_step"),
    ("u_residual_norm", "u_residual_norm_by_step"),
    ("u_hybrid_signal", "u_hybrid_signal_by_step"),
    ("variance_scaled_signal", "variance_scaled_signal_by_step"),
    ("top_book_disagreement", "top_book_disagreement_by_step"),
    ("top_book_residual_norm", "top_book_residual_norm_by_step"),
    ("top_book_hybrid_signal", "top_book_hybrid_signal_by_step"),
)


MODEL_SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = BASE_MODEL_SIGNAL_SPECS

CANONICAL_SIGNAL_TRACE_KEYS: Tuple[str, ...] = tuple(out_key for _, out_key in MODEL_SIGNAL_SPECS) + (
    CANONICAL_INFO_GROWTH_TRACE_KEY,
)


def resolved_r_star(residual_norm_values: Sequence[float]) -> float:
    residual = np.asarray(residual_norm_values, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return 1.0
    return max(float(np.mean(np.clip(residual, 0.0, None))), 1e-8)


def compute_canonical_info_growth_hardness(
    residual_norm,
    disagreement,
    *,
    r_star: float,
):
    if float(r_star) <= 0.0:
        raise ValueError(f"r_star must be positive, got {r_star}")
    return disagreement * torch.log1p(residual_norm / float(r_star))


def compute_canonical_info_growth_hardness_numpy(
    residual_norm: Sequence[float] | np.ndarray,
    disagreement: Sequence[float] | np.ndarray,
    *,
    r_star: float,
) -> np.ndarray:
    if float(r_star) <= 0.0:
        raise ValueError(f"r_star must be positive, got {r_star}")
    residual_arr = np.asarray(residual_norm, dtype=np.float64)
    disagreement_arr = np.asarray(disagreement, dtype=np.float64)
    return disagreement_arr * np.log1p(np.clip(residual_arr, 0.0, None) / float(r_star))


def compute_no_rstar_info_growth_hardness(
    residual_norm,
    disagreement,
):
    return disagreement * torch.log1p(torch.clamp(residual_norm, min=0.0))


def compute_no_rstar_info_growth_hardness_numpy(
    residual_norm: Sequence[float] | np.ndarray,
    disagreement: Sequence[float] | np.ndarray,
) -> np.ndarray:
    residual_arr = np.asarray(residual_norm, dtype=np.float64)
    disagreement_arr = np.asarray(disagreement, dtype=np.float64)
    return disagreement_arr * np.log1p(np.clip(residual_arr, 0.0, None))


__all__ = [
    "BASE_MODEL_SIGNAL_SPECS",
    "CANONICAL_INFO_GROWTH_ROW_KEY",
    "CANONICAL_INFO_GROWTH_TRACE_KEY",
    "CANONICAL_SIGNAL_TRACE_KEYS",
    "EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_ROW_KEY",
    "EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY",
    "MODEL_SIGNAL_SPECS",
    "compute_canonical_info_growth_hardness",
    "compute_canonical_info_growth_hardness_numpy",
    "compute_no_rstar_info_growth_hardness",
    "compute_no_rstar_info_growth_hardness_numpy",
    "resolved_r_star",
]
