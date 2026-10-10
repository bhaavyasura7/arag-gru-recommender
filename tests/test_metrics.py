from arag.eval.metrics import hit_at_k, mrr, ndcg_at_k, rank_of


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
