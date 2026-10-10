"""Paired bootstrap for per-user metric differences (95% CI)."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """95% CI (percentile) of the mean per-user difference a - b.

    Returns (mean_diff, ci_low, ci_high). CI containing 0 => no significant difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b) > 0
    diffs = a - b
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(ci_low), float(ci_high)
