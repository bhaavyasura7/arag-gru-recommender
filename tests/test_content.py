import numpy as np

from arag.ranking.content import ItemKNN, content_ranking, recency_weights


def test_recency_weights_most_recent_is_one():
    w = recency_weights(3, 0.5)
    assert w[-1] == 1.0 and w[0] < w[1] < w[2]
    assert np.allclose(recency_weights(4, 0.0), 1.0)


def test_content_ranking_prefers_similar_items():
    emb = np.eye(3, dtype=np.float32)
    emb_idx = {"a": 0, "b": 1, "c": 2}
    out = content_ranking(["a"], ["c", "b", "a"], emb, emb_idx)
    assert out[0] == "a"
    assert sorted(out) == ["a", "b", "c"]


def test_content_ranking_decay_favours_recent():
    emb = np.eye(2, dtype=np.float32)
    emb_idx = {"x": 0, "y": 1}
    # history x then y; with decay the recent item y dominates
    assert content_ranking(["x", "y"], ["x", "y"], emb, emb_idx, decay=1.0)[0] == "y"


def test_content_ranking_unknown_candidates_last():
    emb = np.eye(2, dtype=np.float32)
    out = content_ranking(["a"], ["zz", "a", "b"], emb, {"a": 0, "b": 1})
    assert out[-1] == "zz"


def test_itemknn_cooccurrence():
    knn = ItemKNN([["a", "b"], ["a", "b"], ["a", "c"], ["d"]])
    out = knn.rank(["a"], ["c", "d", "b", "unseen"])
    assert out[0] == "b" and out[1] == "c"
    assert out[-1] == "unseen"
    assert sorted(out) == sorted(["c", "d", "b", "unseen"])
