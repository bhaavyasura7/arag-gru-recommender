"""Repository layer for the demo memory (SQLite)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd

from arag import util
from arag.config import get_config
from arag.memory.db import connect, init_db

EVENT_TYPES = {"view", "click", "like", "dismiss", "save"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def seed_db(cfg, db_path: str | Path, n_users: int = 8) -> None:
    logger = util.setup_logging("arag.memory.seed")
    conn = init_db(db_path)
    proc = cfg.path("data_processed")
    lists = cfg.path("lists")

    items = pd.read_parquet(proc / "items.parquet")
    interactions = pd.read_parquet(proc / "interactions.parquet")
    users_meta = util.read_json(lists / "users.json")

    # all items with text
    text_items = items[items["has_text"]]
    cur = conn.cursor()
    for _, row in text_items.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO items (item_id, title, text, main_category, price) VALUES (?,?,?,?,?)",
            (str(row["item_id"]), str(row.get("title", "")), str(row.get("text", "")),
             str(row.get("main_category", "")), float(row["price"]) if pd.notna(row.get("price")) else None),
        )

    test_users = users_meta["test"][:n_users]
    for u in test_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at, is_synthetic) VALUES (?,?,0)",
            (u, _now_iso()),
        )
        hist = interactions[
            (interactions["user_id"] == u) & (interactions["split"].isin(["train", "valid"]))
        ].sort_values(["timestamp", "item_id"])
        for _, h in hist.iterrows():
            cur.execute(
                "INSERT INTO interactions (user_id, item_id, ts, event_type, rating, source) VALUES (?,?,?,?,?,?)",
                (u, str(h["item_id"]), int(h["timestamp"]), "view", float(h["rating"]), "dataset"),
            )

    conn.commit()
    conn.close()
    logger.info("seeded %d demo users and %d items", len(test_users), len(text_items))


def list_users(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT user_id, created_at, is_synthetic FROM users ORDER BY user_id").fetchall()
    return [dict(r) for r in rows]


def user_exists(conn: sqlite3.Connection, user_id: str) -> bool:
    return conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is not None


def create_session(conn: sqlite3.Connection, user_id: str, session_id: str | None = None) -> str:
    session_id = session_id or uuid.uuid4().hex
    now = _now_iso()
    conn.execute(
        "INSERT INTO sessions (session_id, user_id, created_at, updated_at) VALUES (?,?,?,?)",
        (session_id, user_id, now, now),
    )
    conn.commit()
    return session_id


def get_session(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def touch_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (_now_iso(), session_id))
    conn.commit()


def add_interaction(conn: sqlite3.Connection, user_id: str, item_id: str, event_type: str,
                    rating: float | None = None, source: str = "app") -> None:
    assert event_type in EVENT_TYPES
    conn.execute(
        "INSERT INTO interactions (user_id, item_id, ts, event_type, rating, source) VALUES (?,?,?,?,?,?)",
        (user_id, item_id, _now_ms(), event_type, rating, source),
    )
    conn.commit()


def add_session_event(conn: sqlite3.Connection, session_id: str, user_id: str, item_id: str,
                      event_type: str, payload: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO session_events (session_id, user_id, item_id, event_type, ts, payload_json) VALUES (?,?,?,?,?,?)",
        (session_id, user_id, item_id, event_type, _now_ms(), json.dumps(payload or {})),
    )
    conn.commit()


def interacted_items(conn: sqlite3.Connection, user_id: str) -> set[str]:
    rows = conn.execute("SELECT DISTINCT item_id FROM interactions WHERE user_id=?", (user_id,)).fetchall()
    return {r["item_id"] for r in rows}


def dismissed_items(conn: sqlite3.Connection, user_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT item_id FROM interactions WHERE user_id=? AND event_type='dismiss'", (user_id,)
    ).fetchall()
    return {r["item_id"] for r in rows}


def save_profile(conn: sqlite3.Connection, user_id: str, profile: dict, provenance: dict, version: int) -> None:
    conn.execute(
        """INSERT INTO profiles (user_id, profile_json, provenance_json, version, updated_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json,
             provenance_json=excluded.provenance_json, version=excluded.version, updated_at=excluded.updated_at""",
        (user_id, json.dumps(profile), json.dumps(provenance), version, _now_iso()),
    )
    conn.commit()


def get_profile(conn: sqlite3.Connection, user_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["profile_json"] = json.loads(d["profile_json"])
    d["provenance_json"] = json.loads(d["provenance_json"])
    return d


def delete_profile(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute("DELETE FROM profiles WHERE user_id=?", (user_id,))
    conn.commit()


def count_app_events(conn: sqlite3.Connection, user_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM interactions WHERE user_id=? AND source='app'", (user_id,)
    ).fetchone()
    return int(row["c"])


def create_run(conn: sqlite3.Connection, user_id: str, session_id: str, kind: str,
               request_json: dict, config_json: dict) -> str:
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO rec_runs (run_id, user_id, session_id, kind, request_json, config_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (run_id, user_id, session_id, kind, json.dumps(request_json), json.dumps(config_json), _now_iso()),
    )
    return run_id


def add_rec_items(conn: sqlite3.Connection, run_id: str, items: list[dict]) -> None:
    for it in items:
        conn.execute(
            "INSERT OR IGNORE INTO rec_items (run_id, item_id, rank, score, reason_json) VALUES (?,?,?,?,?)",
            (run_id, it["item_id"], it["rank"], it["score"], json.dumps(it.get("reason", {}))),
        )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM rec_runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["request_json"] = json.loads(d["request_json"])
    d["config_json"] = json.loads(d["config_json"])
    d["items"] = [
        dict(r) for r in conn.execute(
            "SELECT * FROM rec_items WHERE run_id=? ORDER BY rank", (run_id,)
        ).fetchall()
    ]
    for it in d["items"]:
        it["reason"] = json.loads(it["reason_json"])
    return d
