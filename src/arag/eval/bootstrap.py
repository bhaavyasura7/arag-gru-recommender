"""Paired bootstrap for per-user metric differences (95% CI)."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
    max_n: int = 20000,
    chunk: int = 500,
) -> tuple[float, float, float]:
    """95% CI (percentile) of the mean per-user difference a - b.

    Returns (mean_diff_full, ci_low, ci_high). ``mean_diff_full`` is computed on
    all users; the CI is bootstrapped, subsampling to ``max_n`` users when the set
    is larger (the CI is insensitive to n beyond that and this keeps memory bounded).
    A CI containing 0 => no significant difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b) > 0
    diffs = a - b
    full_mean = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n > max_n:
        diffs = diffs[rng.choice(n, size=max_n, replace=False)]
        n = max_n

    means = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = diffs[idx].mean(axis=1)
        done += m

    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return full_mean, float(ci_low), float(ci_high)
