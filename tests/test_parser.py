from arag.ranking.parser import parse_ranking


def test_valid_ranking_passthrough():
    ids = [f"c{i}" for i in range(20)]
    out, stats = parse_ranking(list(ids), ids)
    assert out == ids
    assert stats.unknown == stats.duplicate == stats.missing == 0


def test_unknown_dropped_and_missing_appended_in_order():
    candidate_ids = ["a", "b", "c", "d"]
    ranking = ["a", "x", "b", "z", "a"]  # x,z unknown; a duplicated; c,d missing
    out, stats = parse_ranking(ranking, candidate_ids)
    assert stats.unknown == 2
    assert stats.duplicate == 1
    assert stats.missing == 2
    assert out == ["a", "b", "c", "d"]
    assert sorted(out) == sorted(candidate_ids)
    assert len(set(out)) == len(out) == 4


def test_missing_appended_by_nli_score():
    candidate_ids = ["a", "b", "c"]
    ranking = ["a"]  # b, c missing
    out, stats = parse_ranking(ranking, candidate_ids, nli_scores={"b": 9, "c": 2})
    assert out == ["a", "b", "c"]
    assert stats.missing == 2


def test_empty_ranking_yields_list_order():
    candidate_ids = ["a", "b", "c"]
    out, stats = parse_ranking([], candidate_ids)
    assert out == ["a", "b", "c"]
    assert stats.missing == 3
