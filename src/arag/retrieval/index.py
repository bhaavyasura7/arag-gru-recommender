"""Build a FAISS IndexFlatIP over the normalised item embeddings."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from arag import util
from arag.config import get_config


def main() -> None:
    logger = util.setup_logging("arag.retrieval.index")
    cfg = get_config()
    faiss_dir: Path = cfg.path("faiss")

    embeddings = np.load(faiss_dir / "item_embeddings.npy")
    item_ids = np.load(faiss_dir / "item_ids.npy", allow_pickle=True)
    d = embeddings.shape[1]

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_dir / "index.faiss"))

    util.write_json(
        faiss_dir / "index_meta.json",
        {"n_items": int(len(item_ids)), "dim": int(d), "index": "IndexFlatIP"},
    )
    logger.info("FAISS index written: %d items, dim=%d", len(item_ids), d)


if __name__ == "__main__":
    main()
