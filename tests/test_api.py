import pytest
from fastapi.testclient import TestClient

from arag.api.app import create_app
from arag.api.services import RecommenderService
from arag.config import load_config
from arag.memory import repo
from arag.memory.db import init_db


class StubService(RecommenderService):
    """Service with heavy retrieval/ranking internals stubbed (no LLM/GRU/FAISS)."""

    def health(self):
        return {"db": True, "faiss": False, "gru": False, "llm_key": True, "model": self.cfg.llm_model}

    def _rank(self, user_id, history_df, candidate_ids):
        return list(candidate_ids), {}

    def _retrieve_feed(self, conn, user_id, history, k=50):
        ids = [r["item_id"] for r in conn.execute("SELECT item_id FROM items").fetchall()]
        return self._filter_candidates(conn, user_id, ids, 20)

    def _retrieve_search(self, conn, user_id, q, filters, k=50):
        ids = [r["item_id"] for r in conn.execute("SELECT item_id FROM items").fetchall()]
        return self._filter_candidates(conn, user_id, ids, k)

    def _parse_query(self, query):
        return {"query": query, "keywords": query.split(), "max_price": None, "exclude_terms": []}

    def _render_items(self, item_ids, reasons):
        conn = repo.connect(self.db_path)
        try:
            rows = {r["item_id"]: dict(r) for r in conn.execute("SELECT * FROM items").fetchall()}
        finally:
            conn.close()
        out = []
        for iid in item_ids:
            row = rows.get(iid)
            out.append({
                "item_id": iid,
                "title": row["title"] if row else iid,
                "category": row["main_category"] if row else "",
                "price": row["price"] if row else None,
                "reason": reasons.get(iid, ""),
                "evidence": "",
                "score": 0.0,
            })
        return out


@pytest.fixture()
def client(tmp_path, ollama_cfg):
    cfg = ollama_cfg
    db = tmp_path / "app.db"
    init_db(db)
    conn = repo.connect(db)
    conn.execute("INSERT INTO users (user_id, created_at, is_synthetic) VALUES ('u1','',0),('u2','',0)")
    conn.execute(
        "INSERT INTO items (item_id, title, text, main_category, price) VALUES "
        "('a','A','alpha','',1.0),('b','B','beta','',2.0),('c','C','gamma','',3.0)"
    )
    conn.commit()
    conn.close()
    svc = StubService(cfg, str(db))
    app = create_app(service=svc, db_path=str(db))
    return TestClient(app)


def _session(client, user="u1"):
    r = client.post("/sessions", json={"user_id": user})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["db"] is True


def test_users(client):
    r = client.get("/users")
    ids = {u["user_id"] for u in r.json()}
    assert ids == {"u1", "u2"}


def test_feed_cross_user_rejected(client):
    sid = _session(client, "u1")
    r = client.get("/feed", params={"user_id": "u2", "session_id": sid})
    assert r.status_code == 403


def test_feedback_persists_and_dismiss_changes_feed(client):
    sid = _session(client, "u1")
    r = client.get("/feed", params={"user_id": "u1", "session_id": sid})
    assert r.status_code == 200
    assert "a" in [i["item_id"] for i in r.json()["items"]]

    r = client.post("/feedback", json={"user_id": "u1", "session_id": sid, "item_id": "a", "event_type": "dismiss"})
    assert r.status_code == 200

    r = client.get("/feed", params={"user_id": "u1", "session_id": sid})
    assert "a" not in [i["item_id"] for i in r.json()["items"]]


def test_profile_delete(client):
    r = client.delete("/users/u1/profile")
    assert r.status_code == 200
    r = client.get("/users/u1/profile")
    assert r.status_code == 404


def test_replay_unknown_run_404(client):
    r = client.post("/runs/nonexistent/replay", json={})
    assert r.status_code == 404


def test_search_requires_valid_session(client):
    r = client.post("/search", json={"user_id": "u1", "session_id": "bad", "query": "x"})
    assert r.status_code == 404
