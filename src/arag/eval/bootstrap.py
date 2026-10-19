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

    Returns (mean_diff, ci_low, ci_high). A CI containing 0 => no significant difference.

    Up to ``max_n`` users: classic index resampling. Above it (all-users runs), the
    resampled means are drawn exactly but cheaply: per-user differences take only a few
    distinct values (ranks 1..20), so a bootstrap sample is a multinomial draw of counts
    over those values. (Before 2026-10-20 large sets were subsampled to ``max_n`` users,
    which centred the CI on the subsample mean and made it ~2x too wide.)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b) > 0
    diffs = a - b
    full_mean = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n > max_n:
        vals, counts = np.unique(diffs, return_counts=True)
        if len(vals) <= 10000:
            draws = rng.multinomial(n, counts / n, size=n_resamples)
            means = draws @ vals / n
            ci_low, ci_high = np.percentile(means, [2.5, 97.5])
            return full_mean, float(ci_low), float(ci_high)
        chunk = max(1, min(chunk, 10_000_000 // n))  # many distinct values: exact, memory-bounded

    means = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = diffs[idx].mean(axis=1)
        done += m

    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return full_mean, float(ci_low), float(ci_high)
