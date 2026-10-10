import pandas as pd
import pytest

from arag.agents.nli import NLIResult, run_nli, select_accepted
from arag.config import load_config
from arag.llm.cache import DiskCache
from arag.llm.client import LLMClient


def make_nli(items: dict[str, int]) -> NLIResult:
    r = NLIResult()
    for iid, score in items.items():
        r.items[iid] = {"label": "neutral", "score": score, "rationale": ""}
    return r


def test_select_accepted_normal():
    nli = make_nli({"a": 8, "b": 6, "c": 3})
    accepted, empty = select_accepted(nli, theta=6)
    assert empty is False
    assert [iid for iid, _ in accepted] == ["a", "b"]


def test_select_accepted_empty_fallback_top3():
    nli = make_nli({"a": 3, "b": 2, "c": 1, "d": 0})
    accepted, empty = select_accepted(nli, theta=6)
    assert empty is True
    assert len(accepted) == 3
    assert [iid for iid, _ in accepted] == ["a", "b", "c"]


@pytest.fixture()
def client(tmp_path, ollama_cfg):
    return LLMClient(ollama_cfg, cache=DiskCache(tmp_path / "cache"), logger=None)


def test_run_nli_covers_all_candidate_ids(client, require_ollama, ollama_cfg):
    candidates = [
        {"item_id": "A1", "title": "a", "text": "a game"},
        {"item_id": "A2", "title": "b", "text": "a movie"},
        {"item_id": "A3", "title": "c", "text": "a book"},
    ]
    result = run_nli(client, ollama_cfg, "user likes games", candidates, per_call=5)
    ids = {c["item_id"] for c in candidates}
    # every candidate ends up with a score (missing -> neutral 5 after re-ask)
    assert set(result.items.keys()) == ids
    for d in result.items.values():
        assert d["label"] in ("support", "neutral", "contradict")
        assert 0 <= d["score"] <= 10
