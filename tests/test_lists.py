import pandas as pd

from arag.dataprep.lists import _build_lists


def make_items(n=100):
    return pd.DataFrame(
        {"item_id": [f"i{k}" for k in range(n)], "has_text": [True] * n}
    )


def make_seq(users=("u1", "u2")):
    rows = []
    for u in users:
        rows.append(
            {
                "user_id": u,
                "train_items": [f"{u}_a", f"{u}_b"],
                "valid_item": f"{u}_v",
                "valid_ts": 100,
                "test_item": f"{u}_t",
                "test_ts": 200,
            }
        )
    return pd.DataFrame(rows)


def test_lists_20_unique_target_once_no_interacted():
    items = make_items(100)
    seq = make_seq(["u1"])
    df = _build_lists(seq, items, "valid", seed=42)
    assert len(df) == 1
    cands = df.iloc[0]["candidates"]
    assert len(cands) == 20
    assert len(set(cands)) == 20
    target = df.iloc[0]["target"]
    assert cands.count(target) == 1
    interacted = {"u1_a", "u1_b", "u1_v", "u1_t"}
    negatives = [c for c in cands if c != target]
    assert not set(negatives) & interacted


def test_lists_byte_identical_regeneration():
    items = make_items(100)
    seq = make_seq(["u1", "u2"])
    a = _build_lists(seq, items, "test", seed=42)
    b = _build_lists(seq, items, "test", seed=42)
    assert a.equals(b)
    assert a["candidates"].tolist() == b["candidates"].tolist()


def test_lists_different_seed_differs():
    items = make_items(100)
    seq = make_seq(["u1"])
    a = _build_lists(seq, items, "test", seed=42)
    b = _build_lists(seq, items, "test", seed=43)
    assert a["candidates"].tolist() != b["candidates"].tolist()
