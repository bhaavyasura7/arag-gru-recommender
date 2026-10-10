"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal["view", "click", "like", "dismiss", "save"]


class SessionCreate(BaseModel):
    user_id: str
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: str


class RecItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float | None = None
    reason: str = ""
    evidence: str = ""
    score: float = 0.0


class FeedResponse(BaseModel):
    run_id: str
    user_id: str
    page: int
    page_size: int
    items: list[RecItem]


class SearchRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    filters: dict[str, Any] | None = None
    limit: int = 5


class SearchResponse(BaseModel):
    run_id: str
    items: list[RecItem]


class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    item_id: str
    event_type: EventType


class OKResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    db: bool
    faiss: bool
    gru: bool
    llm_key: bool
    model: str


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict
    provenance: dict
    version: int
    updated_at: str | None = None


class ErrorResponse(BaseModel):
    detail: str
