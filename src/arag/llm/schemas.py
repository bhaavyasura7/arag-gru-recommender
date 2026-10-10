"""Pydantic output schemas for each agent, plus strict JSON-schema derivation.

The same schemas are used to (a) build the ``response_format`` JSON schema sent to
the model and (b) validate the parsed output locally.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class UUAOutput(BaseModel):
    long_term_interests: list[str]
    current_intent: str
    avoid: list[str]
    summary: str


class NLIItem(BaseModel):
    item_id: str
    label: Literal["support", "neutral", "contradict"]
    score: int = Field(ge=0, le=10)
    rationale: str


class NLIOutput(BaseModel):
    results: list[NLIItem]


class CSAOutput(BaseModel):
    summary: str


class IRAReason(BaseModel):
    item_id: str
    reason: str


class IRAOutput(BaseModel):
    ranking: list[str]
    reasons: list[IRAReason] = Field(default_factory=list)


class RankingOutput(BaseModel):
    ranking: list[str]


class SearchQuery(BaseModel):
    keywords: list[str]
    max_price: float | None = None
    exclude_terms: list[str] = Field(default_factory=list)


def _clean(node: Any) -> Any:
    if isinstance(node, dict):
        node = {k: v for k, v in node.items() if k != "title"}
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        return {k: _clean(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_clean(x) for x in node]
    return node


def to_strict_schema(model_cls: type[BaseModel], name: str) -> dict:
    """Derive a strict JSON schema (no additionalProperties, all fields required)."""
    raw = model_cls.model_json_schema()
    return {"name": name, "strict": True, "schema": _clean(raw)}
