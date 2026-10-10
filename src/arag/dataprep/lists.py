"""Build sequences, user samples, and frozen candidate lists.

Outputs (all deterministic, seed 42):

- ``artifacts/lists/users.json``              100 validation + 100 test user ids
- ``artifacts/lists/sequences.parquet``       per-user train sequence + targets
- ``artifacts/lists/{valid,test}_lists.parquet``  frozen 20-item candidate lists
- ``artifacts/lists/lists_hashes.json``       SHA-256 of the list parquet files

The candidate lists are frozen: once written, re-running must be byte-identical and
must never be regenerated after evaluation starts.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from arag import util
from arag.config import get_config

SPLITS = ("valid", "test")


def _group_sequence(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_seq, targets) where targets has one row per user with
    valid_item/valid_ts/test_item/test_ts and train_last_ts."""
    train = interactions[interactions["split"] == "train"].sort_values(
        ["user_id", "timestamp", "item_id"]
    )
    train_seq = (
        train.groupby("user_id")["item_id"].apply(list).rename("train_items").reset_index()
    )
    train_last = (
        train.groupby("user_id")["timestamp"].max().rename("train_last_ts").reset_index()
    )
    train_seq = train_seq.merge(train_last, on="user_id", how="left")

    targets = train_seq[["user_id"]].copy()
    for split in SPLITS:
        sub = interactions[interactions["split"] == split].sort_values(
            ["user_id", "timestamp", "item_id"]
        )
        agg = (
            sub.groupby("user_id")
            .agg(item=("item_id", "last"), ts=("timestamp", "max"))
            .rename(columns={"item": f"{split}_item", "ts": f"{split}_ts"})
            .reset_index()
        )
        targets = targets.merge(agg, on="user_id", how="left")
    return train_seq, targets


def _assert_no_leak(seq: pd.DataFrame, logger) -> None:
    bad = 0
    for _, row in seq.iterrows():
        train_items = row["train_items"]
        valid_item = row["valid_item"]
        test_item = row["test_item"]
        valid_ts = row["valid_ts"]
        test_ts = row["test_ts"]
        train_last_ts = row.get("train_last_ts")
        if pd.isna(valid_item) or pd.isna(test_item):
            continue
        history_for_valid = set(train_items)
        history_for_test = history_for_valid | {valid_item}
        if valid_item in history_for_valid or test_item in history_for_test:
            bad += 1
        if train_last_ts is not None and not pd.isna(train_last_ts) and train_last_ts > valid_ts:
            bad += 1
        if valid_ts > test_ts:
            bad += 1
    if bad:
        logger.warning("users violating target-not-in-history ordering: %d", bad)
    # hard assert: targets must not be in their own history
    assert bad == 0, "target leakage detected in leave-last-out sequences"


def _negative_pool(
    items: pd.DataFrame,
    popularity: dict[str, int] | None,
    strategy: str,
) -> tuple[list[str], list[float] | None]:
    """Return (pool, cumulative weights or None for uniform sampling).

    Without ``popularity`` the pool is every item with text (unit tests only).
    With it, the pool is restricted to items that have train interactions: items
    nobody reviewed are trivially rejected by Popularity and are OOV for the GRU,
    which made uniform-over-catalogue lists far too easy for those two systems.
    ``strategy="popularity"`` samples proportionally to train interaction count, so
    negatives are about as popular as targets and popularity alone cannot separate them.
    """
    text_ids = items.loc[items["has_text"], "item_id"].astype(str)
    if popularity is None:
        return sorted(text_ids.tolist()), None
    pool = sorted(i for i in text_ids if popularity.get(i, 0) > 0)
    if strategy == "uniform":
        return pool, None
    if strategy != "popularity":
        raise ValueError(f"unknown negative sampling strategy: {strategy}")
    cum, total = [], 0.0
    for i in pool:
        total += popularity[i]
        cum.append(total)
    return pool, cum


def _build_lists(
    seq: pd.DataFrame,
    items: pd.DataFrame,
    split: str,
    seed: int,
    popularity: dict[str, int] | None = None,
    strategy: str = "popularity",
    n_negatives: int = 19,
) -> pd.DataFrame:
    catalogue_set = set(items.loc[items["has_text"], "item_id"].astype(str))
    pool, cum_weights = _negative_pool(items, popularity, strategy)
    rng = random.Random(seed)

    rows = []
    for _, row in seq.iterrows():
        user_id = row["user_id"]
        target = row[f"{split}_item"]
        if pd.isna(target):
            continue
        target = str(target)
        interacted = set(row["train_items"]) | {
            str(row["valid_item"]),
            str(row["test_item"]),
        }
        negatives: list[str] = []
        seen: set[str] = set()
        while len(negatives) < n_negatives:
            k = (n_negatives - len(negatives)) + 20
            if cum_weights is None:
                draw = rng.sample(pool, min(k, len(pool)))
            else:
                draw = rng.choices(pool, cum_weights=cum_weights, k=k)
            for c in draw:
                if c not in interacted and c not in seen:
                    seen.add(c)
                    negatives.append(c)
                    if len(negatives) >= n_negatives:
                        break
        candidates = negatives + [target]
        rng.shuffle(candidates)
        target_has_text = target in catalogue_set
        rows.append(
            {
                "user_id": user_id,
                "split": split,
                "target": target,
                "candidates": candidates,
                "target_has_text": target_has_text,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logger = util.setup_logging("arag.dataprep.lists")
    cfg = get_config()
    proc_dir: Path = cfg.path("data_processed")
    lists_dir: Path = cfg.path("lists")
    lists_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed

    interactions = pd.read_parquet(proc_dir / "interactions.parquet")
    items = pd.read_parquet(proc_dir / "items.parquet")

    train_seq, targets = _group_sequence(interactions)
    seq = train_seq.merge(targets, on="user_id", how="left")
    seq = seq.sort_values("user_id").reset_index(drop=True)
    seq["n_train"] = seq["train_items"].apply(len)
    logger.info("users: %d (with valid+test target: %d)",
                len(seq), int(seq[["valid_item", "test_item"]].notna().all(axis=1).sum()))
    _assert_no_leak(seq, logger)

    seq.to_parquet(lists_dir / "sequences.parquet", index=False)

    # user samples: 100 validation + 100 test users whose target has text
    text_ids = set(items.loc[items["has_text"], "item_id"].astype(str))
    samples: dict[str, list[str]] = {}
    rng = random.Random(seed)
    for split in SPLITS:
        eligible = seq[seq[f"{split}_item"].astype(str).isin(text_ids)]
        n_key = "sampling.n_validation_users" if split == "valid" else "sampling.n_test_users"
        picked = sorted(rng.sample(eligible["user_id"].tolist(), cfg.get(n_key)))
        samples[split] = picked
        logger.info("%s users sampled: %d", split, len(picked))
    util.write_json(
        lists_dir / "users.json",
        {"seed": seed, "validation": samples["valid"], "test": samples["test"]},
    )

    train = interactions[interactions["split"] == "train"]
    popularity = train["item_id"].astype(str).value_counts().to_dict()
    strategy = cfg.get("sampling.negative_sampling", "popularity")
    n_negatives = cfg.get("sampling.n_negatives", 19)
    logger.info("negative sampling: %s over %d items with train interactions",
                strategy, len(popularity))

    hashes = {}
    for split in SPLITS:
        lists_df = _build_lists(seq, items, split, seed, popularity, strategy, n_negatives)
        path = lists_dir / f"{split}_lists.parquet"
        lists_df.to_parquet(path, index=False)
        hashes[f"{split}_lists.parquet"] = util.sha256_file(path)
        logger.info("%s lists written: %d users", split, len(lists_df))
    util.write_json(lists_dir / "lists_hashes.json", hashes)
    logger.info("lists hashes: %s", hashes)


if __name__ == "__main__":
    main()
