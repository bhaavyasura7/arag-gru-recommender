from arag.ranking.fusion import fuse


def test_alpha_1_reproduces_llm_order():
    ids = [f"c{i}" for i in range(20)]
    llm = ids[::-1]          # reverse order
    gru = ids                # identity order
    out = fuse(ids, llm, gru, alpha=1.0)
    assert out == llm


def test_alpha_0_reproduces_gru_order():
    ids = [f"c{i}" for i in range(20)]
    llm = ids[::-1]
    gru = ids
    out = fuse(ids, llm, gru, alpha=0.0)
    assert out == gru


def test_fusion_is_permutation():
    ids = [f"c{i}" for i in range(20)]
    llm = ids[::-1]
    gru = ids
    out = fuse(ids, llm, gru, alpha=0.5)
    assert sorted(out) == sorted(ids)
    assert len(set(out)) == len(out) == 20


def test_ties_broken_by_nli_then_id():
    ids = ["a", "b", "c"]
    llm = ["a", "b", "c"]
    gru = ["c", "b", "a"]  # s_gru: c=1.0, b=0.5, a=0.0
    # alpha=0.5: final = 0.5*s_llm + 0.5*s_gru
    # a: 0.5*1.0 + 0.5*0.0 = 0.5 ; b: 0.5*0.5 + 0.5*0.5 = 0.5 ; c: 0.5*0.0 + 0.5*1.0 = 0.5
    # all tie -> NLI score breaks it
    out = fuse(ids, llm, gru, alpha=0.5, nli_scores={"a": 1, "b": 3, "c": 2})
    assert out == ["b", "c", "a"]
