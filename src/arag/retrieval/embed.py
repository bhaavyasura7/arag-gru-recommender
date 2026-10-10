"""Embed item text with sentence-transformers (normalised) and save to disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from arag import util
from arag.config import get_config


def main() -> None:
    logger = util.setup_logging("arag.retrieval.embed")
    cfg = get_config()
    proc_dir: Path = cfg.path("data_processed")
    faiss_dir: Path = cfg.path("faiss")
    faiss_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(proc_dir / "items.parquet")
    text_items = items[items["has_text"]].reset_index(drop=True)
    logger.info("items with text to embed: %d", len(text_items))

    model = SentenceTransformer(cfg.get("embedding.model"))
    texts = text_items["text"].astype(str).tolist()
    logger.info("embedding %d texts ...", len(texts))
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(faiss_dir / "item_ids.npy", text_items["item_id"].astype(str).to_numpy())
    np.save(faiss_dir / "item_embeddings.npy", embeddings)
    logger.info("saved %d embeddings (%s) -> %s", len(embeddings), embeddings.shape, faiss_dir)


if __name__ == "__main__":
    main()
