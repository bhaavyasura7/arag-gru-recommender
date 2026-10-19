import numpy as np

from arag.eval.bootstrap import paired_bootstrap


def test_paired_bootstrap_all_better():
    a = [0.8] * 50
    b = [0.3] * 50
    mean_diff, lo, hi = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    assert mean_diff > 0.4
    assert lo > 0 and hi > 0  # clearly significant


def test_paired_bootstrap_identical():
    a = [0.5] * 50
    b = [0.5] * 50
    mean_diff, lo, hi = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    assert abs(mean_diff) < 1e-9
    assert lo <= 0 <= hi  # CI contains 0 -> not significant


def test_paired_bootstrap_large_n_uses_all_users():
    # above max_n the CI must be centred on the full mean and match index resampling
    rng = np.random.default_rng(0)
    r = rng.integers(1, 21, 30000)
    a = np.where(r <= 5, 1 / np.log2(r + 1), 0.0)
    b = np.roll(a, 7) * 0.9
    m, lo, hi = paired_bootstrap(a, b, n_resamples=2000, max_n=1000)
    _, lo_ref, hi_ref = paired_bootstrap(a, b, n_resamples=2000, max_n=10**9)
    assert lo < m < hi
    assert abs(lo - lo_ref) < 0.001 and abs(hi - hi_ref) < 0.001
