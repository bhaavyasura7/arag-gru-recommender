from arag.llm.schemas import NLIOutput, UUAOutput, to_strict_schema


def test_strict_schema_is_closed_and_required():
    schema = to_strict_schema(UUAOutput, "uua")
    s = schema["schema"]
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"].keys())
    assert "title" not in s


def test_strict_schema_handles_nested_defs():
    schema = to_strict_schema(NLIOutput, "nli")
    s = schema["schema"]
    assert "$defs" in s
    item_schema = s["$defs"]["NLIItem"]
    assert item_schema["additionalProperties"] is False
    assert set(item_schema["required"]) == set(item_schema["properties"].keys())
