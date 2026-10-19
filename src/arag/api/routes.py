"""API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from arag.api.schemas import (
    FeedResponse,
    FeedbackRequest,
    OKResponse,
    ProfileResponse,
    SearchRequest,
    SearchResponse,
    SessionCreate,
    SessionResponse,
)

router = APIRouter()


def _service(request: Request):
    return request.app.state.service


@router.get("/health")
def health(request: Request):
    return _service(request).health()


@router.get("/dataset")
def dataset(request: Request):
    return _service(request).dataset_stats()


@router.get("/users")
def users(request: Request):
    return _service(request).list_users()


@router.post("/sessions", response_model=SessionResponse)
def create_session(request: Request, body: SessionCreate):
    try:
        return _service(request).create_session(body.user_id, body.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/feed", response_model=FeedResponse)
def feed(
    request: Request,
    user_id: str,
    session_id: str,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=50),
):
    try:
        return _service(request).feed(user_id, session_id, page, page_size)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/search", response_model=SearchResponse)
def search(request: Request, body: SearchRequest):
    try:
        return _service(request).search(body.user_id, body.session_id, body.query, body.filters, body.limit)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/feedback", response_model=OKResponse)
def feedback(request: Request, body: FeedbackRequest):
    try:
        _service(request).feedback(body.user_id, body.session_id, body.item_id, body.event_type)
        return {"ok": True}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/users/{user_id}/history")
def history(request: Request, user_id: str, limit: int = Query(30, ge=1, le=100)):
    return {"user_id": user_id, "items": _service(request).history(user_id, limit)}


@router.get("/users/{user_id}/profile", response_model=ProfileResponse)
def get_profile(request: Request, user_id: str):
    p = _service(request).get_profile(user_id)
    if p is None:
        raise HTTPException(status_code=404, detail="no profile")
    return {
        "user_id": user_id,
        "profile": p["profile_json"],
        "provenance": p["provenance_json"],
        "version": p["version"],
        "updated_at": p["updated_at"],
    }


@router.post("/users/{user_id}/profile", response_model=ProfileResponse)
def generate_profile(request: Request, user_id: str):
    try:
        p = _service(request).generate_profile(user_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "user_id": user_id,
        "profile": p["profile_json"],
        "provenance": p["provenance_json"],
        "version": p["version"],
        "updated_at": p["updated_at"],
    }


@router.delete("/users/{user_id}/profile", response_model=OKResponse)
def delete_profile(request: Request, user_id: str):
    _service(request).delete_profile(user_id)
    return {"ok": True}


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str):
    run = _service(request).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.post("/runs/{run_id}/replay")
def replay(request: Request, run_id: str):
    try:
        return _service(request).replay(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/results")
def results(request: Request):
    import os
    from arag.config import get_config
    cfg = get_config()
    p = cfg.path("results") / "metrics.json"
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="results/metrics.json not found")
    import json
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)
