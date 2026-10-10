import pandas as pd

from arag.agents.context import (
    build_context,
    history_before,
    render_item_line,
)


def test_render_item_line_format():
    line = render_item_line("Some Title", "Video Games", 4.5, 1600000000000)
    assert line.startswith("Some Title | Video Games | 4.5 | ")
    assert line.endswith("2020-09-13")


def test_build_context_splits_session_and_longterm():
    hist = pd.DataFrame(
        [
            {"item_id": f"i{k}", "title": f"T{k}", "main_category": "Video Games",
             "rating": 4.0, "timestamp": 1000 + k}
            for k in range(10)
        ]
    ).sort_values(["timestamp", "item_id"])
    c_st, c_lt = build_context(hist, n_session=5, n_long=50)
    assert len(c_st) == 5
    assert len(c_lt) == 5
    assert c_st[0].startswith("T5")
    assert c_lt[-1].startswith("T4")


def test_history_before_excludes_target_and_later():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "a", "timestamp": 100, "rating": 5},
            {"user_id": "u1", "item_id": "t", "timestamp": 300, "rating": 5},  # target
            {"user_id": "u1", "item_id": "b", "timestamp": 200, "rating": 5},
            {"user_id": "u1", "item_id": "c", "timestamp": 400, "rating": 5},  # after target
            {"user_id": "u1", "item_id": "d", "timestamp": 300, "rating": 5},  # same time
        ]
    )
    hist = history_before(df, target_ts=300, target_id="t")
    ids = hist["item_id"].tolist()
    assert ids == ["a", "b"]
    assert "t" not in ids
    assert "c" not in ids
    assert "d" not in ids
