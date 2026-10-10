import pytest
from pydantic import BaseModel, model_validator

from arag.config import load_config
from arag.llm.cache import DiskCache
from arag.llm.client import LLMClient, LLMError
from arag.llm.schemas import NLIOutput, to_strict_schema


@pytest.fixture()
def client(tmp_path):
    cfg = load_config()
    return LLMClient(cfg, cache=DiskCache(tmp_path / "cache"), logger=None)


def test_complete_json_cache_second_run_zero_calls(client, require_ollama):
    schema = to_strict_schema(NLIOutput, "nli")
    prompt = "USER HISTORY: bought a Nintendo Switch. CANDIDATES: B001 a Switch game."
    first = client.complete_json("nli", prompt, schema)
    assert isinstance(first, dict)
    assert len(client.records) == 1
    assert client.records[0].cache_hit is False

    # second identical run must hit cache: zero new API calls
    second = client.complete_json("nli", prompt, schema)
    assert second == first
    assert len(client.records) == 2
    assert client.records[1].cache_hit is True
    assert sum(1 for r in client.records if not r.cache_hit) == 1


def test_complete_validated_nli(client, require_ollama):
    schema = to_strict_schema(NLIOutput, "nli")
    prompt = (
        "USER HISTORY: bought a Nintendo Switch and Mario games.\n"
        "CANDIDATES:\nB001 a Zelda game for Switch\nB002 a kitchen blender"
    )
    res = client.complete_validated("nli", prompt, schema, NLIOutput)
    # Plumbing check: strict schema was honoured -> no fallback, valid type.
    assert res.fallback is False
    assert isinstance(res.obj, NLIOutput)
    for it in res.obj.results:
        assert isinstance(it.item_id, str) and it.item_id
        assert it.label in ("support", "neutral", "contradict")
        assert 0 <= it.score <= 10


class _AlwaysFails(BaseModel):
    x: int

    @model_validator(mode="after")
    def _fail(self):
        raise ValueError("deliberately failing validator for test")


def test_complete_validated_retries_then_uses_fallback(client, require_ollama):
    schema = to_strict_schema(_AlwaysFails, "fails")
    fallback = {"x": 0}
    res = client.complete_validated("test", "return {\"x\": 1}", schema, _AlwaysFails, fallback=fallback)
    assert res.fallback is True
    assert res.obj == fallback
    # initial call + one retry = 2 API calls
    assert len(client.records) == 2
    assert all(not r.cache_hit for r in client.records)


def test_replay_only_refuses_uncached_call(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLAY_ONLY", "1")
    cfg = load_config()
    c = LLMClient(cfg, cache=DiskCache(tmp_path / "cache"), logger=None)
    schema = to_strict_schema(NLIOutput, "nli")
    with pytest.raises(LLMError):
        c.complete_json("nli", "some brand new uncached prompt", schema)
