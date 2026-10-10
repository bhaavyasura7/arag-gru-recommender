import logging

import pandas as pd
import pytest

from arag.dataprep.lists import _assert_no_leak, _group_sequence


def make_interactions(rows):
    return pd.DataFrame(
        rows, columns=["user_id", "item_id", "timestamp", "rating", "split"]
    )


def test_leave_last_out_ordering_and_no_leak():
    rows = [
        ("u1", "a", 100, 5, "train"),
        ("u1", "b", 200, 4, "train"),
        ("u1", "c", 300, 3, "train"),
        ("u1", "v", 400, 5, "valid"),
        ("u1", "t", 500, 5, "test"),
        ("u2", "x", 100, 5, "train"),
        ("u2", "y", 200, 5, "train"),
        ("u2", "v2", 300, 4, "valid"),
        ("u2", "t2", 400, 5, "test"),
    ]
    interactions = make_interactions(rows)
    train_seq, targets = _group_sequence(interactions)
    seq = train_seq.merge(targets, on="user_id", how="left")

    assert len(seq) == 2
    for _, row in seq.iterrows():
        assert row["train_last_ts"] <= row["valid_ts"] <= row["test_ts"]
        assert row["valid_item"] not in row["train_items"]
        assert row["test_item"] not in row["train_items"] + [row["valid_item"]]

    _assert_no_leak(seq, logging.getLogger())


def test_leak_detected_when_test_item_in_train_history():
    rows = [
        ("u1", "a", 100, 5, "train"),
        ("u1", "t", 200, 5, "train"),  # t leaks: also the test target
        ("u1", "v", 300, 5, "valid"),
        ("u1", "t", 400, 5, "test"),
    ]
    interactions = make_interactions(rows)
    train_seq, targets = _group_sequence(interactions)
    seq = train_seq.merge(targets, on="user_id", how="left")
    with pytest.raises(AssertionError):
        _assert_no_leak(seq, logging.getLogger())


def test_group_sequence_takes_last_per_split():
    rows = [
        ("u1", "a", 100, 5, "train"),
        ("u1", "b", 200, 5, "train"),
        ("u1", "v_old", 250, 5, "valid"),
        ("u1", "v_new", 300, 5, "valid"),
        ("u1", "t", 400, 5, "test"),
    ]
    interactions = make_interactions(rows)
    train_seq, targets = _group_sequence(interactions)
    seq = train_seq.merge(targets, on="user_id", how="left")
    row = seq.iloc[0]
    assert row["valid_item"] == "v_new"
    assert row["valid_ts"] == 300
