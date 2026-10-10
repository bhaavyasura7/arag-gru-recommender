"""Build session (C_st) and long-term (C_lt) context strings for prompts.

Context is built only from history strictly before the target (the caller passes a
history frame that already excludes the target and any later interactions).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def render_item_line(title: str, main_category: str, rating: float, ts_ms: int) -> str:
    date = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{title} | {main_category} | {rating:g} | {date}"


def render_lines(history: pd.DataFrame) -> list[str]:
    """Render a sorted history frame (item_id,title,main_category,rating,timestamp)."""
    lines = []
    for _, row in history.iterrows():
        lines.append(
            render_item_line(
                str(row.get("title", "")),
                str(row.get("main_category", "")),
                float(row.get("rating", 0.0)),
                int(row.get("timestamp", 0)),
            )
        )
    return lines


def build_context(
    history: pd.DataFrame, n_session: int = 5, n_long: int = 50
) -> tuple[list[str], list[str]]:
    """Return (c_st, c_lt) line lists from a history frame sorted ascending.

    ``history`` must be sorted by (timestamp, item_id) ascending and contain only
    interactions strictly before the target.
    """
    n = len(history)
    st_start = max(0, n - n_session)
    lt_start = max(0, st_start - n_long)
    c_st = render_lines(history.iloc[st_start:n])
    c_lt = render_lines(history.iloc[lt_start:st_start])
    return c_st, c_lt


def history_before(
    interactions: pd.DataFrame, target_ts: int, target_id: str | None = None
) -> pd.DataFrame:
    """Return rows strictly before ``target_ts`` (and excluding ``target_id``).

    Sorted ascending by (timestamp, item_id). This is the only function allowed to
    select history for a prompt; it guarantees no leakage of the target or anything
    at/after the target's time.
    """
    hist = interactions[interactions["timestamp"] < target_ts]
    if target_id is not None:
        hist = hist[hist["item_id"].astype(str) != str(target_id)]
    return hist.sort_values(["timestamp", "item_id"])
