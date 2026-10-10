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
