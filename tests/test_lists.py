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


def test_lists_negatives_only_from_items_with_train_interactions():
    items = make_items(100)
    seq = make_seq(["u1", "u2"])
    # only i0..i39 were ever interacted with in train
    popularity = {f"i{k}": k + 1 for k in range(40)}
    df = _build_lists(seq, items, "test", seed=42, popularity=popularity, strategy="uniform")
    for _, row in df.iterrows():
        negatives = [c for c in row["candidates"] if c != row["target"]]
        assert len(negatives) == 19
        assert all(c in popularity for c in negatives)


def test_lists_popularity_sampling_follows_train_counts():
    items = make_items(100)
    seq = make_seq([f"u{k}" for k in range(200)])
    # i0 is 50x more popular than every other item
    popularity = {f"i{k}": (500 if k == 0 else 10) for k in range(60)}
    df = _build_lists(seq, items, "test", seed=42, popularity=popularity, strategy="popularity")
    share_i0 = sum("i0" in c for c in df["candidates"]) / len(df)
    share_i1 = sum("i1" in c for c in df["candidates"]) / len(df)
    assert share_i0 > 0.9
    assert share_i1 < 0.5
    a = _build_lists(seq, items, "test", seed=42, popularity=popularity)
    assert a["candidates"].tolist() == df["candidates"].tolist()
