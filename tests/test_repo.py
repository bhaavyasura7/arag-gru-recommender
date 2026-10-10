import pytest

from arag.memory import repo
from arag.memory.db import init_db


@pytest.fixture()
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    c.execute("INSERT INTO users (user_id, created_at, is_synthetic) VALUES ('u1','',0),('u2','',0)")
    c.execute(
        "INSERT INTO items (item_id, title, text, main_category, price) VALUES "
        "('a','A','alpha','',1.0),('b','B','beta','',2.0),('c','C','gamma','',3.0)"
    )
    c.commit()
    yield c
    c.close()


def test_feedback_persists(conn):
    repo.add_interaction(conn, "u1", "a", "like")
    assert "a" in repo.interacted_items(conn, "u1")


def test_dismiss_excludes(conn):
    repo.add_interaction(conn, "u1", "b", "dismiss")
    assert repo.dismissed_items(conn, "u1") == {"b"}


def test_profile_save_get_delete(conn):
    repo.save_profile(conn, "u1", {"summary": "x"}, {"version": 1}, 1)
    p = repo.get_profile(conn, "u1")
    assert p["profile_json"]["summary"] == "x"
    repo.delete_profile(conn, "u1")
    assert repo.get_profile(conn, "u1") is None


def test_session_create_and_get(conn):
    sid = repo.create_session(conn, "u1")
    s = repo.get_session(conn, sid)
    assert s["user_id"] == "u1"


def test_run_roundtrip(conn):
    run_id = repo.create_run(conn, "u1", "s1", "feed", {"page": 0}, {"alpha": 0.5})
    repo.add_rec_items(conn, run_id, [{"item_id": "a", "rank": 1, "score": 0.9, "reason": {}}])
    run = repo.get_run(conn, run_id)
    assert run["kind"] == "feed"
    assert run["items"][0]["item_id"] == "a"
