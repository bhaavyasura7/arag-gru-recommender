"""Load raw files into Parquet tables: ``interactions`` and ``items``.

Logs row counts at every step; drops rows with missing ids/timestamps and logs
how many (no silent row loss). Updates the manifest with row counts and writes the
tracked copy to ``results/manifest.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arag import util
from arag.config import get_config
from arag.dataprep import manifest as manifest_mod
from arag.dataprep.items import build_item_text, parse_price

INTERACTION_COLUMNS = ["user_id", "item_id", "timestamp", "rating", "split"]


def _read_benchmark(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"user_id": str, "parent_asin": str},
    )
    df = df.rename(columns={"parent_asin": "item_id"})
    df["split"] = split
    return df[INTERACTION_COLUMNS]


def _load_items(meta_path: Path, max_words: int, logger) -> pd.DataFrame:
    rows: list[dict] = []
    missing_id = 0
    with open(meta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed metadata line")
                continue
            item_id = meta.get("parent_asin")
            if item_id is None or str(item_id).strip() == "":
                missing_id += 1
                continue
            title = meta.get("title")
            text = build_item_text(meta, max_words=max_words)
            rows.append(
                {
                    "item_id": str(item_id),
                    "title": str(title).strip() if title else "",
                    "text": text,
                    "main_category": (
                        str(meta.get("main_category")).strip()
                        if meta.get("main_category")
                        else ""
                    ),
                    "price": parse_price(meta.get("price")),
                    "has_text": bool(title) and bool(text),
                }
            )
    if missing_id:
        logger.info("metadata rows with missing parent_asin dropped: %d", missing_id)
    items = pd.DataFrame(rows)
    before = len(items)
    items = items.drop_duplicates(subset="item_id", keep="first")
    logger.info("items metadata rows: %d -> %d after dedup", before, len(items))
    return items


def main() -> None:
    logger = util.setup_logging("arag.dataprep.load")
    cfg = get_config()
    raw_dir: Path = cfg.path("data_raw")
    proc_dir: Path = cfg.path("data_processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    category = cfg.category

    # --- interactions -----------------------------------------------------
    frames = []
    for split in ("train", "valid", "test"):
        path = raw_dir / f"{category}.{split}.csv"
        df = _read_benchmark(path, split)
        logger.info("%s rows read: %d", split, len(df))
        frames.append(df)
    interactions = pd.concat(frames, ignore_index=True)

    n_raw = len(interactions)
    interactions = interactions.dropna(subset=["user_id", "item_id", "timestamp"])
    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()
    interactions["item_id"] = interactions["item_id"].astype(str).str.strip()
    interactions = interactions[
        (interactions["user_id"] != "") & (interactions["item_id"] != "")
    ]
    dropped = n_raw - len(interactions)
    if dropped:
        logger.info("rows dropped for missing user/item/timestamp: %d", dropped)

    interactions["timestamp"] = interactions["timestamp"].astype("int64")
    interactions["rating"] = interactions["rating"].astype(float)
    interactions = interactions[INTERACTION_COLUMNS].reset_index(drop=True)
    interactions.to_parquet(proc_dir / "interactions.parquet", index=False)
    logger.info("interactions written: %d rows", len(interactions))

    # --- items ------------------------------------------------------------
    meta_path = raw_dir / f"meta_{category}.jsonl"
    items = _load_items(meta_path, cfg.get("splits.item_text_words"), logger)
    items.to_parquet(proc_dir / "items.parquet", index=False)
    logger.info(
        "items written: %d rows (%d with text)",
        len(items),
        int(items["has_text"].sum()),
    )

    # items referenced by interactions but absent from metadata
    interacted_ids = set(interactions["item_id"].unique())
    known_ids = set(items["item_id"].unique())
    no_meta = interacted_ids - known_ids
    logger.info("interacted items with no metadata/text: %d", len(no_meta))

    # --- manifest ---------------------------------------------------------
    manifest = manifest_mod.load(cfg)
    manifest["rows"] = {
        "interactions": int(len(interactions)),
        "items": int(len(items)),
        "items_with_text": int(items["has_text"].sum()),
        "users": int(interactions["user_id"].nunique()),
        "interacted_items_without_metadata": len(no_meta),
    }
    for split in ("train", "valid", "test"):
        manifest["rows"][f"interactions_{split}"] = int(
            (interactions["split"] == split).sum()
        )
    manifest_mod.save(cfg, manifest)
    manifest_mod.write_tracked_copy(cfg, manifest)
    logger.info("manifest updated + tracked copy written")


if __name__ == "__main__":
    main()
