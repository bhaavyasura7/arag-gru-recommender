from arag.eval.metrics import hit_at_k, mrr, ndcg_at_k, rank_of
from arag.eval.run import compute_all_metrics


class _Cfg:
    def get(self, key, default=None):
        return default


def test_rank_of():
    ranking = ["a", "b", "c"]
    assert rank_of("a", ranking) == 1
    assert rank_of("c", ranking) == 3
    assert rank_of("z", ranking) == 4


def test_hit_at_k():
    assert hit_at_k(1, 5) == 1.0
    assert hit_at_k(5, 5) == 1.0
    assert hit_at_k(6, 5) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(1, 5) == 1.0  # 1/log2(2) = 1
    assert ndcg_at_k(3, 5) == 0.5  # 1/log2(4) = 0.5
    assert ndcg_at_k(6, 5) == 0.0


def test_mrr():
    assert mrr(1) == 1.0
    assert mrr(2) == 0.5
    assert mrr(4) == 0.25


def test_compute_all_metrics_subset_without_reference_systems():
    # regression: the non-LLM "all users" block passes systems that contain
    # neither 'arag' nor every reference; pairwise refs must be skipped, not KeyError
    systems = {
        "random": {"u1": ["a", "b", "c"], "u2": ["b", "a", "c"]},
        "popularity": {"u1": ["b", "a", "c"], "u2": ["a", "b", "c"]},
        "gru_seed0": {"u1": ["a", "b", "c"], "u2": ["a", "b", "c"]},
    }
    targets = {"u1": "a", "u2": "b"}
    out = compute_all_metrics(systems, targets, _Cfg())
    assert out["random"]["hit@5"]["mean"] == 1.0
    assert "popularity_vs_gru_seed0" in out  # reference present in subset
    assert not any(k.startswith("random_vs_arag") for k in out)  # absent ref skipped
