"""SQLite schema and connection management for the demo memory."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  created_at TEXT,
  is_synthetic INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS items (
  item_id TEXT PRIMARY KEY,
  title TEXT,
  text TEXT,
  main_category TEXT,
  price REAL
);
CREATE TABLE IF NOT EXISTS interactions (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  item_id TEXT,
  ts INTEGER,
  event_type TEXT,
  rating REAL,
  source TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_ts ON interactions(user_id, ts);
CREATE TABLE IF NOT EXISTS profiles (
  user_id TEXT PRIMARY KEY,
  profile_json TEXT,
  provenance_json TEXT,
  version INTEGER,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS session_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  user_id TEXT,
  item_id TEXT,
  event_type TEXT,
  ts INTEGER,
  payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_events_sess_ts ON session_events(session_id, ts);
CREATE TABLE IF NOT EXISTS rec_runs (
  run_id TEXT PRIMARY KEY,
  user_id TEXT,
  session_id TEXT,
  kind TEXT,
  request_json TEXT,
  config_json TEXT,
  trace_json TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS rec_items (
  run_id TEXT,
  item_id TEXT,
  rank INTEGER,
  score REAL,
  reason_json TEXT,
  PRIMARY KEY (run_id, item_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(rec_runs)")}
    if "trace_json" not in cols:
        conn.execute("ALTER TABLE rec_runs ADD COLUMN trace_json TEXT")
