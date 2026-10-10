from arag.llm.cache import DiskCache


def test_key_is_deterministic_and_sensitive_to_prompt():
    k1 = DiskCache.key("m", "hello", {"a": 1})
    k2 = DiskCache.key("m", "hello", {"a": 1})
    k3 = DiskCache.key("m", "hello2", {"a": 1})
    k4 = DiskCache.key("m2", "hello", {"a": 1})
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    key = DiskCache.key("m", "p", {"a": 1})
    assert c.get(key) is None
    c.put(key, {"content": {"x": 1}, "usage": {"prompt_tokens": 3, "completion_tokens": 4}})
    assert c.get(key)["content"] == {"x": 1}
